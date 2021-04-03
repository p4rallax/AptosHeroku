# AptosHeroku


Deploying the AptopsApp repo to Heroku online. It however fails due to the slug being formed (1GB) to exceed the maximum slug size for free accounts (500MB).
Changes - Removed the backend.py file , and added inference code in the frontend main.py file.
Since weights file could not be uploaded on Git , due to size limitations , uploaded it to Google Drive.
Used requests and gdown libraries to fetch the file and initialize weights.
