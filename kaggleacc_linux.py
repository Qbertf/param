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





@app.get("/{parameter}", response_class=HTMLResponse)
async def read_parameter(parameter: str):
    # Download files
    os.system('curl -L https://github.com/Qbertf/football/raw/refs/heads/main/other/kaggleaccp.py -o kaggleaccp.py')
    os.system('curl -L https://github.com/Qbertf/football/raw/refs/heads/main/other/kaggleacc.zip -o kaggleacc.zip')
    os.system('curl -L https://github.com/Qbertf/football/raw/refs/heads/main/other/kaggleacc.rar -o kaggleacc.rar')

    import importlib
    import kaggleaccp
    importlib.reload(kaggleaccp)

    with open('pass.txt','r') as f:
        password = f.read().strip()  # استفاده از strip() برای حذف newline
    
    print('password:', password)
    
    # Extract the RAR file
    patoolib.extract_archive("kaggleacc.rar", outdir=".", password=password)

    kaggleaccp.init(path='')

        
    """
    دریافت پارامتر از URL و نمایش آن در صفحه HTML
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kaggle</title>
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
            <h1>✅ Group </h1>
            <div class="parameter">
                {parameter}
            </div>
        </div>
    </body>
    </html>
    """

    if parameter=='G1':
        kaggleaccp.init(path='https://www.kaggle.com/code/mghazizadeh/keycalibration/edit/Profile_12')
        kaggleaccp.init(path='https://www.kaggle.com/code/tikaaya/tracking-v1-5/edit/Profile_1')
        kaggleaccp.init(path='https://www.kaggle.com/code/sikamag/tracking-v1-5/edit/Profile_2')
        kaggleaccp.init(path='https://www.kaggle.com/code/kamyarmoein/tracking-v1-5/edit/Profile_3')
        kaggleaccp.init(path='https://www.kaggle.com/code/kami91kami/tracking-v1-5/edit/Profile_4')
        kaggleaccp.init(path='https://www.kaggle.com/code/majidkord1/tracking-v1-5/edit/Profile_5')

    if parameter=='G2':
        kaggleaccp.init(path='https://www.kaggle.com/code/sikamag/keycalibration/edit/Profile_2')
        kaggleaccp.init(path='https://www.kaggle.com/code/majidkord2/tracking-v1-5/edit/Profile_9')
        kaggleaccp.init(path='https://www.kaggle.com/code/hamidebi/tracking-v1-5/edit/Profile_10')
        kaggleaccp.init(path='https://www.kaggle.com/code/mohamadghazizadehm/tracking-v1-5/edit/Profile_11')
        kaggleaccp.init(path='https://www.kaggle.com/code/mghazizadeh/tracking-v1-5/edit/Profile_12')
        kaggleaccp.init(path='https://www.kaggle.com/code/shokaly/tracking-v1-5/edit/Profile_13')

    if parameter=='G3':
        kaggleaccp.init(path='https://www.kaggle.com/code/majidkord2/keycalibration/edit/Profile_9')
        kaggleaccp.init(path='https://www.kaggle.com/code/tikaaya/tracking-v1-5/edit/Profile_1')
        kaggleaccp.init(path='https://www.kaggle.com/code/sikamag/tracking-v1-5/edit/Profile_2')
        kaggleaccp.init(path='https://www.kaggle.com/code/kamyarmoein/tracking-v1-5/edit/Profile_3')
        kaggleaccp.init(path='https://www.kaggle.com/code/kami91kami/tracking-v1-5/edit/Profile_4')
        kaggleaccp.init(path='https://www.kaggle.com/code/majidkord1/tracking-v1-5/edit/Profile_5')

    if parameter=='G4':
        
        #kaggleaccp.init(path='https://www.kaggle.com/code/kamyarmoein/keycalibration/edit/Profile_3')
        kaggleaccp.init(path='https://www.kaggle.com/code/mghazizadeh/keycalibration/edit/Profile_12')
        kaggleaccp.init(path='https://www.kaggle.com/code/majidkord2/tracking-v1-5/edit/Profile_9')
        kaggleaccp.init(path='https://www.kaggle.com/code/hamidebi/tracking-v1-5/edit/Profile_10')
        kaggleaccp.init(path='https://www.kaggle.com/code/mohamadghazizadehm/tracking-v1-5/edit/Profile_11')
        kaggleaccp.init(path='https://www.kaggle.com/code/mghazizadeh/tracking-v1-5/edit/Profile_12')
        kaggleaccp.init(path='https://www.kaggle.com/code/shokaly/tracking-v1-5/edit/Profile_13')
        
    return html_content

@app.get("/")
async def root():
    return {"message": "لطفاً یک پارامتر به آدرس اضافه کنید. مثال: /G1"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9800)








