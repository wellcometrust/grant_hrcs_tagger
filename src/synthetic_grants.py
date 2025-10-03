import argparse

import pandas as pd
import boto3

from functools import partial
from langchain_aws import ChatBedrock
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
from tqdm import tqdm


class Grant(BaseModel):
    """Research Grant Proposal"""
    title: str = Field(description="The title of the research grant")
    abstract: str = Field(description="A brief summary of the research grant")


class Grants(BaseModel):
    """Collection of Research Grant Proposals"""
    grants: List[Grant] = Field(description="List of research grant proposals")


class GrantGenerator:
    def __init__(self, train_path, text_path, criteria_path, llm_provider, **model_params):

        self.llm = self.setup_llm(llm_provider, **model_params)
        self.generation_chain = self.setup_generation_chain()
        self.check_chain = self.setup_check_chain()

        self.train_df = self.load_train_df(train_path, text_path)
        self.criteria = self.load_criteria(criteria_path)

    @property
    def grant_generation_prompt_template(self):
        return PromptTemplate.from_template(
            "You are a helpful assistant that creates research grants based on a set of criteria. "
            "Each criterion comes with a title and definition, as well as a parent title and definition. "
            "Here is a list of all possible selection criteria for reference:\n{criteria}\n"
            "Create {num_grants} research grant proposals based ONLY on the following selection criteria:\n{codes}.\n"
            "Make the grants as thematically varied as possible, but be sure to follow the criteria closely. "
            "Each proposal should ONLY consist of a title and an abstract. "
            "Here are some example proposals: {examples}\n"
            "Each abstract should be no more than 300 tokens long.\n"
            "Do NOT refer to the criteria in the proposals. "
            "Do NOT include any justification or explanation of how/why the proposal meets the given criteria.\n"
            "Format your response as a JSON object with a single key 'grants' that maps to a list of objects, each with a 'title' and 'abstract' field.\n"
            "Follow this output format precisely. Do NOT include any other text in your response.\n"
        )
    
    @property
    def grant_check_prompt_template(self):
        return PromptTemplate.from_template(
            "You are a helpful assistant that validates research grants against a set of criteria. "
            "Each criterion comes with a title and definition, as well as a parent title and definition.\n"
            "RULES:\n"
            "For each grant, respond with 'yes' if the grant meets all the criteria, and 'no' if it does not. "
            "Do NOT include any justification or explanation of how/why the proposal meets the given criteria.\n"
            "ONLY respond with 'yes' or 'no'. Do NOT include any other text in your response.\n"
            "GRANT:\nTitle:{title}\nAbstract:{abstract}\n"
            "CRITERIA:\n{criteria}\n"
            "Does the grant meet the criteria? "
        )
    
    @staticmethod
    def setup_bedrock(**model_kwargs):
        bedrock_client = boto3.client(service_name='bedrock-runtime')
        llm = ChatBedrock(
            client=bedrock_client,
            region_name='eu-west-1',
            **model_kwargs
        ).with_structured_output(Grants)
        return llm
    
    @staticmethod
    def setup_ollama(**model_kwargs):
        return ChatOllama(**model_kwargs)

    def setup_llm(self, llm_provider, **model_kwargs):
        if llm_provider == "bedrock":
            return self.setup_bedrock(**model_kwargs)
        
        elif llm_provider == "ollama":
            return self.setup_ollama(**model_kwargs)

        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    def setup_generation_chain(self):
        chain = self.grant_generation_prompt_template | self.llm
        return chain
    
    def setup_check_chain(self):
        chain = self.grant_check_prompt_template | self.llm
        return chain
    
    @staticmethod
    def load_train_df(train_path, text_path) -> pd.DataFrame:
        train_df = pd.read_parquet(train_path)
        text_df = pd.read_parquet(text_path)
        text_df['text'] = text_df['AllText'].str.lower()
        train_df = train_df.merge(
            text_df[['text', 'AwardTitle', 'AwardAbstract']], on='text', how='inner'
            )
        train_df.dropna(inplace=True)
        train_df['composite'] = train_df['RA'].apply(lambda x: ' '.join(sorted(x)))
        return train_df

    @staticmethod
    def load_criteria(criteria_path):
        criteria_df = pd.read_excel(criteria_path)
        criteria = {}
        for code, criteria in criteria_df.set_index('child_category').to_dict(orient='index').items():
            criteria_str = f"""
            Parent Title: {criteria['parent_name'].replace("\n"," ")}
            Parent Definition: {criteria['parent_definition'].replace("\n"," ")}
            Title: {criteria['child_name'].replace("\n"," ")}
            Definition: {criteria['inclusion_criteria'].replace("\n"," ")}
            """
            criteria[code] = criteria_str

        return criteria

    @property
    def category_counts(self):
        return self.train_df['composite'].value_counts()

    def get_examples(self, composite_category, n_examples = 3):
        n_examples = min(n_examples, self.category_counts.get(composite_category, 0))
        examples_df = self.train_df[self.train_df['composite'] == composite_category].sample(n_examples)
        titles = examples_df['AwardTitle'].tolist()
        abstracts = examples_df['AwardAbstract'].tolist()
        grants = []
        for t, a in zip(titles, abstracts):
            grants.append(
                Grant(
                    title=t.replace('\r','').replace('\n',''), 
                    abstract=a.replace('\r','').replace('\n','')
                )
            )

        return Grants(grants=grants)
    
    def generate(self, criteria, category, n_grants, n_examples):
        synthetic_grants = self.generation_chain.invoke(
            {
                "criteria": criteria,
                "codes": category,
                "examples": self.get_examples(category, n_examples=n_examples).model_dump_json(),
                "num_grants": n_grants
            }
        )

        if not hasattr(synthetic_grants, 'grants') and hasattr(synthetic_grants, 'content'):
            synthetic_grants = Grants.model_validate_json(synthetic_grants.content)

        return synthetic_grants

    def generate_grants(self, max_n_grants=1000, max_categories=100, n_examples=3, batchsize=10, dump_path=None):
        single_codes = [code for code in self.train_df['composite'].unique() if ' ' not in code]

        for category, count in pd.concat([
                self.category_counts[single_codes], self.category_counts.nlargest(max_categories)
            ]).drop_duplicates().items():

            if (count >= max_n_grants):
                continue
            n_grants = max_n_grants - count
            
            categories = category.split(' ')
            criteria = '\n'.join([self.criteria[cat] for cat in categories if cat in self.criteria])
            if len(categories) > 1:
                n_grants = n_grants // len(categories)
            
            print(f"Generating {n_grants} grants for category: {category}")

            for i in tqdm(range(0, n_grants, batchsize)):
                try:
                    synthetic_grants = self.generate(criteria, category, min(batchsize, n_grants - i), n_examples)

                    if dump_path:
                        with open(dump_path, 'a') as f:
                            for grant in synthetic_grants.grants:
                                f.write(f"{category}\t{grant.title}\t{grant.abstract}\n")
                except Exception as e:
                    print(f"Error generating grants for category {category} at batch {i}: {e}")
                    continue

    def check_grants(self, grants_df):
        checked = []
        for row in tqdm(grants_df.itertuples()):
            title = row.title
            abstract = row.abstract
            categories = row.RA
            criteria = '\n'.join([self.criteria[cat] for cat in categories if cat in self.criteria])

            try:
                response = self.check_chain.invoke(
                    {
                        "criteria": criteria,
                        "title": title,
                        "abstract": abstract
                    }
                )
                checked.append(response.strip().lower() == 'yes')
            
            except Exception as e:
                checked.append(None)

        return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--text_path", type=str, required=True)
    parser.add_argument("--criteria_path", type=str, required=True)
    parser.add_argument("--llm_provider", type=str, choices=["bedrock", "ollama"], required=True)
    parser.add_argument("--model_id", type=str, default="eu.anthropic.claude-sonnet-4-20250514-v1:0")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=3500)
    parser.add_argument("--max_categories", type=int, default=100)
    parser.add_argument("--n_examples", type=int, default=3)
    parser.add_argument("--max_n_grants", type=int, default=500)
    parser.add_argument("--dump_path", type=str, default=None)
    args = parser.parse_args()


    grant_generator = partial(GrantGenerator, 
        train_path=args.train_path,
        text_path=args.text_path,
        criteria_path=args.criteria_path,
        llm_provider=args.llm_provider,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    if args.llm_provider == "bedrock":
        grant_generator = grant_generator(model_id=args.model_id)
    elif args.llm_provider == "ollama":
        grant_generator = grant_generator(model = args.model_id)

    grant_generator.generate_grants(max_n_grants=args.max_n_grants, dump_path=args.dump_path, max_categories=args.max_categories,
       n_examples=args.n_examples)
