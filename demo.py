import requests


class BLC_Demo:
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
    
    def display_result(self, result):
        """Display prediction result"""
        print("\nPREDICTION:")
        print(f"Language: {'ITALIAN' if result['is_italian'] else 'NON-ITALIAN'}")
        print(f"Confidence: {result['confidence']:.1%}")
    
    
    def run_demo(self):
        """Main demo loop"""
        print("\nBINARY LANGUAGE CLASSIFIER - DEMO\n")
        
        # Check server status
        print("Checking API server status...")
        if not self.check_server_status():
            print("ERROR: API server is not running!")
            print("\nPlease start the server first:")
            print("  python main.py")
            print("\nThen run this demo script again.")
            return
        print("API server is running and model is loaded!")
        
        # Show initial instructions
        print("\nINSTRUCTIONS:")
        print("- Enter any text to classify as Italian or Non-Italian")
        print("- Type 'clear' to clear the screen")
        print("- Type 'exit' or press Enter with empty text to end the demo")
        
        while True:
            try:
                # Get user input
                text = input("\nEnter text to classify: ").strip()
                
                # Handle special commands
                if not text or text.lower() == 'exit':
                    print("Thanks for using the Binary Language Classifier!")
                    print("Demo ended.")
                    break
                
                if text.lower() == 'clear':
                    import os
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                # Make prediction
                print("\nProcessing...")
                result, error = self.predict_text(text)
                
                if error:
                    print(f"\nError: {error}")
                    continue
                
                # Display result
                self.display_result(result)
                
                # Ask if user wants to continue
                print("\nPress Enter to continue or type 'exit' to quit:")
                continue_input = input().strip().lower()
                if continue_input == 'exit':
                    print("Thanks for using the Binary Language Classifier!")
                    print("Demo ended.")
                    break
                    
            except KeyboardInterrupt:
                print("\n\nDemo interrupted by user.")
                print("Thanks for using the Binary Language Classifier!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                print("Please try again or restart the demo.")

def main():
    demo = BLC_Demo()
    demo.run_demo()

if __name__ == "__main__":
    main()