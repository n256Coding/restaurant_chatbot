# restaurant_chatbot
https://ocean-bay.streamlit.app

### Prerequesites
- Open Anaconda Prompt
- Create an anaconda environment.
    ```
    conda create --name com727-chatbot python=3.10
    ```
- Activate the "com727-chatbot" environmment.
    ```
    conda activate com727-chatbot
    ```
- Install dependencies.
    ```
    cd restaurant_chatbot
    pip install -r requirements.txt
    ```

### How to run locally?
- Open Anaconda Prompt
- Activate the "com727-chatbot" environment
- Clone the repository locally (Skip this step if you have already done)
- Move to the project folder
    ```
    cd restaurant_chatbot
    ```
- Download the fasttext model from following location (Skip this step if you have already done)
    https://ssu-my.sharepoint.com/:u:/g/personal/2senan77_solent_ac_uk/EdtaClmEOeFElK4Jb_qAC00BIFrjuF1JCK5kz8n-tEnLww?e=7v738K
    and move to the `data` folder inside the project
- Run the application
    ```
    streamlit run application.py
    ```