from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os  # Make sure this import is at the top
import patoolib
import rarfile

# --- App Initialization ---
app = FastAPI()

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_rar(rar_path, output_dir, password=None):
    """
    Extract RAR archive with optional password
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open RAR file
        with rarfile.RarFile(rar_path) as rf:
            # Set password if provided
            if password:
                rf.setpassword(password)
            
            # Extract all files
            rf.extractall(output_dir)
            print(f"Successfully extracted to {output_dir}")
            return True
            
    except rarfile.RarCannotExec:
        print("Error: unrar tool not found. Install unrar first.")
    except rarfile.BadRarFile:
        print("Error: Corrupted RAR file")
    except rarfile.RarWrongPassword:
        print("Error: Wrong password")
    except Exception as e:
        print(f"Error: {e}")
    
    return False

@app.get("/{parameter}", response_class=HTMLResponse)
async def read_parameter(parameter: str):
    # Download files
    os.system('curl -L https://github.com/Qbertf/football/raw/refs/heads/main/other/kaggleaccp.py -o kaggleaccp.py')
    os.system('curl -L https://github.com/Qbertf/football/raw/refs/heads/main/other/kaggleacc.zip -o kaggleacc.zip')
    os.system('curl -L https://github.com/Qbertf/football/raw/refs/heads/main/other/kaggleacc.rar -o kaggleacc.rar')

    # Extract the RAR file
    extract_rar("kaggleacc.rar", "extracted_files", "1371web3")

    """
    دریافت پارامتر از URL و نمایش آن در صفحه HTML
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>دریافت پارامتر</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                direction: rtl;
                text-align: center;
                padding-top: 50px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                border-radius: 10px;
                padding: 30px;
                max-width: 500px;
                margin: 0 auto;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
            }}
            .parameter {{
                font-size: 24px;
                color: #007bff;
                font-weight: bold;
                margin: 20px 0;
                padding: 15px;
                background-color: #e9f7fe;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>✅ پارامتر دریافت شد</h1>
            <div class="parameter">
                {parameter}
            </div>
            <p>مقدار پارامتر ارسالی در URL نمایش داده شد</p>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/")
async def root():
    return {"message": "لطفاً یک پارامتر به آدرس اضافه کنید. مثال: /G1"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9800)
