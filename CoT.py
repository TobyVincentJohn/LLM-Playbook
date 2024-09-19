import re
from huggingface_hub import InferenceClient
import json

class LLMProcessor:
    def __init__(self, model_name, token="YOUR_HUGGINGFACE_TOKEN"):
        self.client = InferenceClient(model_name, token=token)

    def process_prompt_COT(self,user_query, max_tokens=1000, json_mode=False):
        
        prompt = f"""
        Your task is to break down the thought process needed to answer the following query into a structured, step-by-step Chain of Thought (CoT). Do not provide a direct answer; instead, create a detailed plan outlining each step required to arrive at the final answer. Include substeps where necessary to explain intermediate steps.

        Query: {user_query}

        Expected CoT JSON format:
        {{
            "step 1": "Description of step 1",
            "substeps of 1": {{
                "substep 1.1": "Description of substep 1.1",
                "substep 1.2": "Description of substep 1.2"
            }},
            "step 2": "Description of step 2",
            "substeps of 2": {{
                "substep 2.1": "Description of substep 2.1"
            }},
            "step 3": "Description of step 3",
            "substeps of 2": {{
                "substep 2.1": "Description of substep 2.1"
            }}       
        }}

        Make sure to include all necessary steps and substeps to fully address the query in a clear, logical order while
        keeping it concise.
        """
        out_message = ""
        try:
            for message in self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=True,
            ):
                out_message += message.choices[0].delta.content
        except Exception as e:
            return f"An error occurred while processing the prompt: {e}"

        if json_mode:
            pattern = r"\{.*\}"
            matches = re.findall(pattern, out_message, re.DOTALL)
            
            if matches:
                json_text = matches[-1]  
                try:
                    json_data = json.loads(json_text)
                    return json_data
                except json.JSONDecodeError:
                    return "Extracted text is not valid JSON."
            else:
                return "No JSON found in the output."
        else:
            return out_message


    def process_prompt(self,prompt,max_tokens=800):
        out_message = ""
        try:
            for message in self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=True,
            ):
                out_message += message.choices[0].delta.content
        except Exception as e:
            return f"An error occurred while processing the prompt: {e}"    
        return out_message




    def get_json_values(self, json_data):
        if isinstance(json_data, dict):
            return self._extract_values(json_data)
        else:
            return "Invalid JSON data."

    def _extract_values(self, json_dict, parent_key=''):
        items = []
        for key, value in json_dict.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._extract_values(value, new_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    items.extend(self._extract_values({f"{new_key}[{i}]": item}, ''))
            else:
                items.append((new_key, value))
        return items
    def answer_with_cot(self, user_query):
        cot_json = self.process_prompt_COT(user_query, json_mode=True)
        
        if not isinstance(cot_json, dict):
            return "Failed to generate a valid CoT."
        
        answer_prompt = f"""
        Use the following Chain of Thought (CoT) to answer the query step-by-step:

        CoT: {json.dumps(cot_json, indent=2)}

        Now, based on this CoT, provide the final answer to the query: {user_query}
        """
        return self.process_prompt(answer_prompt)
    def clean_text(self,text):
    # Regex pattern to capture text from the first '*' to the end of the string
        pattern = r'\*(.*)$'
            
            # Find the match
        match = re.search(pattern, text, re.DOTALL)
            
        if match:
            extracted_text = match.group(1)
                
                # Remove all asterisks from the extracted text
            cleaned_text = extracted_text.replace('*', '')
                
            return cleaned_text
if __name__ == "__main__":
    processor = LLMProcessor("meta-llama/Meta-Llama-3-8B-Instruct")
    

    user_query=input()
    print(processor.clean_text(processor.answer_with_cot(user_query)))
    # Use the below piece of code to debug 
    # json_data = processor.process_prompt_COT(user_query, json_mode=False)
    
    # if isinstance(json_data, dict):
    #     print("IT IS JSON")
    #     values = processor.get_json_values(json_data)
    #     for key, value in values:
    #         print(f"{key}: {value}")
    # else:
    #     print(json_data)