#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged Model Testing Program
Test the merged fine-tuned Qwen2.5-7B-Instruct model
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import json
import re

class MergedModelTester:
    def __init__(self):
        """Initialize merged model tester"""
        # 从HF库加载模型
        self.merged_model_path = "WangDong2017/GrammarSeeker-SFT-Qwen2.5-7B"
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load merged fine-tuned model and tokenizer"""
        try:
            # Load merged model and tokenizer directly
            self.model = AutoModelForCausalLM.from_pretrained(
                self.merged_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.merged_model_path,
                trust_remote_code=True
            )
            
            return True
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            return False
    
    def generate_response(self, prompt, max_new_tokens=128):
        """Generate model response"""
        try:
            # Build input format (according to training format)
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Encode input
            inputs = self.tokenizer(input_text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def check_answer(self, model_output, expected_answer):
        """Check if model output contains correct answer"""
        # Clean output text
        output_clean = model_output.lower().strip()
        expected_clean = expected_answer.lower().strip()
        
        # Check if model output contains expected answer
        return expected_clean in output_clean
    
    def load_test_data(self, csv_path):
        """Load test data"""
        try:
            df = pd.read_csv(csv_path)
            
            # Check required columns
            required_columns = ['prompt2', 'answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Error: Missing required columns: {missing_columns}")
                return None
            
            return df
            
        except Exception as e:
            print(f"Failed to load test data: {e}")
            return None
    
    def run_test(self, df):
        """Run test"""
        test_results = []
        correct_count = 0
        positive_correct = 0
        negative_correct = 0
        positive_total = 0
        negative_total = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Test Progress"):
            prompt = row['prompt2']
            expected_answer = row['answer']
            
            # Generate response
            model_output = self.generate_response(prompt)
            
            # Check answer
            is_correct = self.check_answer(model_output, expected_answer)
            
            # Statistics
            if is_correct:
                correct_count += 1
            
            # Determine positive or negative examples based on answer content
            if expected_answer == "T":
                positive_total += 1
                if is_correct:
                    positive_correct += 1
            elif expected_answer == "F":
                negative_total += 1
                if is_correct:
                    negative_correct += 1
            
            # Save result
            result = {
                'index': idx,
                'prompt': prompt,
                'expected_answer': expected_answer,
                'model_output': model_output,
                'is_correct': is_correct
            }
            test_results.append(result)
        
        # Calculate accuracy
        total_accuracy = correct_count / len(df) if len(df) > 0 else 0
        positive_accuracy = positive_correct / positive_total if positive_total > 0 else 0
        negative_accuracy = negative_correct / negative_total if negative_total > 0 else 0
        
        # Calculate F1 score
        precision = positive_correct / (positive_correct + (negative_total - negative_correct)) if (positive_correct + (negative_total - negative_correct)) > 0 else 0
        recall = positive_correct / positive_total if positive_total > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        summary = {
            'f1_score': f1_score,
            'total_samples': len(df),
            'total_correct': correct_count,
            'total_accuracy': total_accuracy,
            'positive_total': positive_total,
            'positive_correct': positive_correct,
            'positive_accuracy': positive_accuracy,
            'negative_total': negative_total,
            'negative_correct': negative_correct,
            'negative_accuracy': negative_accuracy
        }
        
        return test_results, summary
    
    def save_results(self, test_results, summary, test_dataset_name):
        """Save test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "Qwen2.5-7B-Instruct-Merged"
        
        # Create report folder
        report_dir = "report"
        os.makedirs(report_dir, exist_ok=True)
        
        # Save JSON results
        json_filename = os.path.join(report_dir, f"{model_name}_{test_dataset_name}_{timestamp}.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'results': test_results
            }, f, ensure_ascii=False, indent=2)
        
        # Save CSV results
        csv_filename = os.path.join(report_dir, f"{model_name}_{test_dataset_name}_{timestamp}.csv")
        df_results = pd.DataFrame(test_results)
        df_results.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        print(f"\nResults saved to {report_dir} folder:")
        print(f"- JSON: {json_filename}")
        print(f"- CSV: {csv_filename}")
    
    def print_summary(self, summary):
        """Print test summary"""
        print(f"F1 Score: {summary['f1_score']:.4f}")
        print(f"Positive Accuracy: {summary['positive_accuracy']:.4f}")
        print(f"Negative Accuracy: {summary['negative_accuracy']:.4f}")

def main():
    """Main function"""
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
    
    # Create tester
    tester = MergedModelTester()
    
    # Load model
    if not tester.load_model():
        print("Model loading failed, exiting program")
        return
    
    # Check test data file
    test_data_path = "GM-TestData.csv"
    if not os.path.exists(test_data_path):
        print(f"Error: Test data file {test_data_path} does not exist")
        print("Please ensure GM-TestData.csv file is in the current directory")
        return
    
    # Load test data
    df = tester.load_test_data(test_data_path)
    if df is None:
        return
    
    # Run test
    test_results, summary = tester.run_test(df)
    
    # Print results
    tester.print_summary(summary)
    
    # Save results
    tester.save_results(test_results, summary, "GM-TestData")

if __name__ == "__main__":
    main() 