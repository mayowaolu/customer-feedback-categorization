import argparse
import os
import subprocess
import threading
import time

def start_mlflow():
    """Start the MLflow tracking server"""
    print("Starting MLflow tracking server...")
    os.makedirs("mlflow_artifacts", exist_ok=True)
    subprocess.run([
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlflow_artifacts",
        "--host", "0.0.0.0",
        "--port", "5000"
    ])

def start_fastapi():
    """Start the FastAPI service"""
    print("Starting FastAPI service...")
    subprocess.run(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"])

def start_gradio():
    """Start the Gradio interface"""
    print("Starting Gradio interface...")
    subprocess.run(["python", "gradio_app.py"])

def main():
    parser = argparse.ArgumentParser(description="Customer Feedback Classifier Platform")
    parser.add_argument("--mlflow", action="store_true", help="Start MLflow tracking server")
    parser.add_argument("--api", action="store_true", help="Start FastAPI service")
    parser.add_argument("--gradio", action="store_true", help="Start Gradio interface")
    parser.add_argument("--all", action="store_true", help="Start all services")
    
    args = parser.parse_args()
    
    if args.all:
        args.mlflow = args.api = args.gradio = True
    
    if not (args.mlflow or args.api or args.gradio):
        parser.print_help()
        return
    
    threads = []
    
    if args.mlflow:
        mlflow_thread = threading.Thread(target=start_mlflow)
        mlflow_thread.daemon = True
        threads.append(mlflow_thread)
        mlflow_thread.start()
    
    if args.api:
        # Wait a bit if MLflow is starting to ensure it's up before the API tries to connect
        if args.mlflow:
            time.sleep(3)
        
        api_thread = threading.Thread(target=start_fastapi)
        api_thread.daemon = True
        threads.append(api_thread)
        api_thread.start()
    
    if args.gradio:
        gradio_thread = threading.Thread(target=start_gradio)
        gradio_thread.daemon = True
        threads.append(gradio_thread)
        gradio_thread.start()
    
    # Keep the main thread alive to allow Ctrl+C to work properly
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\nShutting down services...")

if __name__ == "__main__":
    main()