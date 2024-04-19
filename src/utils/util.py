# No use here
API_URL = "https://uai0gg1o5neqj4xh.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": f"Bearer {hf_key}",
	"Content-Type": "application/json" 
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


# def load_env():
#     load_dotenv(override=True)
#     hf_key = os.getenv("HF_KEY")
#     openai_key = os.getenv("OPENAI_KEY")
#     client = OpenAI(api_key=openai_key)
#     index = create_database()
#     return hf_key, oepnai_key, client, index