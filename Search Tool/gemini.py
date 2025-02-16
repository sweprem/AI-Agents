



from google import genai

client = genai.Client(api_key="AIzaSyAmN7zbJiyjx8n6KXm7ocmXyqSXsyxHdZA")
response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works"
)
print(response.text)