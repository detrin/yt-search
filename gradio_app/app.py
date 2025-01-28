import gradio as gr
import requests
import time
from requests.auth import HTTPBasicAuth

API_URL = "http://backend:8000"
USERNAME = "admin"
PASSWORD = "secret"

def process_question(question):
    try:
        # Submit search request
        response = requests.post(
            f"{API_URL}/v1/search",
            json={"question": question},
            auth=HTTPBasicAuth(USERNAME, PASSWORD),
            timeout=10
        )
        response.raise_for_status()
        job_id = response.json()["job_id"]
        
        # Poll for status
        while True:
            status_response = requests.get(
                f"{API_URL}/v1/status/{job_id}",
                timeout=10
            )
            status_response.raise_for_status()
            status = status_response.json()["status"]
            
            if status == "done":
                result_response = requests.get(
                    f"{API_URL}/v1/results/{job_id}",
                    timeout=10
                )
                result_response.raise_for_status()
                return result_response.json().get("result", "No result found")
            elif status == "failed":
                return "Processing failed"
            
            time.sleep(1)

    except requests.exceptions.RequestException as e:
        return f"Request error: {str(e)}"
    except KeyError as e:
        return f"Missing key in response: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def app():
    with gr.Blocks() as gradio_app:
        gr.Markdown("# YouTube Search")
        gr.Markdown("Ask questions and get answers from the RAG system")
        
        with gr.Row():
            question_input = gr.Textbox(label="Enter your question", placeholder="Type your question here...")
            answer_output = gr.Textbox(label="Answer", interactive=False)
        
        submit_button = gr.Button("Submit")
        
        submit_button.click(
            fn=process_question,
            inputs=question_input,
            outputs=answer_output
        )
    
    return gradio_app

if __name__ == "__main__":
    app().launch(server_name="0.0.0.0")