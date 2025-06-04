#!/usr/bin/env python3
"""
Interactive Demo Script for Italian Language Classifier
iGenius Take Home Assessment

This script provides an interactive command-line interface to test
the Italian language classifier API.

Usage:
1. Start the API server: python main.py
2. In another terminal, run: python demo.py
"""

import requests
import json
import time
import sys

class ItalianClassifierDemo:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        self.predict_endpoint = f"{api_url}/predict"
        self.health_endpoint = f"{api_url}/health"
    
    def check_server_status(self):
        """Check if the API server is running"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('model_trained', False)
            return False
        except requests.exceptions.RequestException:
            return False
    
    def predict_text(self, text):
        """Send prediction request to API"""
        try:
            response = requests.post(
                self.predict_endpoint,
                json={"Text": text},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json(), None
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                return None, error_msg
                
        except requests.exceptions.RequestException as e:
            return None, f"Connection error: {str(e)}"
    
    def display_result(self, result, text):
        """Display prediction result in a nice format"""
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Input text: '{text}'")
        print(f"Language: {'🇮🇹 ITALIAN' if result['is_italian'] else '🌍 NON-ITALIAN'}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Raw prediction: {result['prediction']} (1=Italian, 0=Non-Italian)")
        
        # Show probability breakdown
        probs = result['probabilities']
        print(f"\nProbability breakdown:")
        print(f"  Italian:     {probs['italian']:.1%}")
        print(f"  Non-Italian: {probs['non_italian']:.1%}")
        print("="*60)
    
    def show_examples(self):
        """Show some example sentences for testing"""
        examples = {
            "Italian examples": [
                "Ciao, come stai?",
                "Mi piace molto la pizza italiana",
                "Buongiorno signora, come va?",
                "Questa è una frase in italiano",
                "Vorrei un caffè per favore",
                "La pasta è molto buona",
                "Dove si trova la stazione?"
            ],
            "Non-Italian examples": [
                "Hello, how are you?",
                "I love Italian food very much",
                "Good morning, how is it going?",
                "This is an English sentence",
                "Would you like some coffee please?",
                "The pasta is very good",
                "Where is the train station?"
            ]
        }
        
        print("\n" + "="*60)
        print("EXAMPLE SENTENCES FOR TESTING")
        print("="*60)
        
        for category, sentences in examples.items():
            print(f"\n{category}:")
            for i, sentence in enumerate(sentences, 1):
                print(f"  {i}. {sentence}")
        
        print("\n" + "="*60)
    
    def run_demo(self):
        """Main demo loop"""
        print("🇮🇹 Italian Language Classifier - Interactive Demo")
        print("=" * 60)
        print("iGenius Take Home Assessment")
        print("=" * 60)
        
        # Check server status
        print("Checking API server status...")
        if not self.check_server_status():
            print("❌ ERROR: API server is not running!")
            print("\nPlease start the server first:")
            print("  python main.py")
            print("\nThen run this demo script again.")
            return
        
        print("✅ API server is running and model is loaded!")
        
        # Show initial instructions
        print("\nINSTRUCTIONS:")
        print("- Enter any text to classify as Italian or Non-Italian")
        print("- Type 'examples' to see example sentences")
        print("- Type 'clear' to clear the screen")
        print("- Type 'quit' or 'exit' to end the demo")
        print("- Press Enter with empty text to quit")
        
        while True:
            try:
                # Get user input
                print("\n" + "-"*60)
                text = input("Enter text to classify: ").strip()
                
                # Handle special commands
                if not text or text.lower() in ['quit', 'exit']:
                    print("\n👋 Thanks for using the Italian Language Classifier!")
                    print("Demo ended.")
                    break
                
                if text.lower() == 'examples':
                    self.show_examples()
                    continue
                
                if text.lower() == 'clear':
                    import os
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                # Make prediction
                print("\n🔄 Processing...")
                result, error = self.predict_text(text)
                
                if error:
                    print(f"\n❌ Error: {error}")
                    continue
                
                # Display result
                self.display_result(result, text)
                
                # Ask if user wants to continue
                print("\nPress Enter to continue (or type 'quit' to exit)")
                continue_input = input().strip().lower()
                if continue_input in ['quit', 'exit']:
                    print("\n👋 Thanks for using the Italian Language Classifier!")
                    print("Demo ended.")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted by user.")
                print("Thanks for using the Italian Language Classifier!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("Please try again or restart the demo.")

def main():
    """Main function"""
    demo = ItalianClassifierDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()