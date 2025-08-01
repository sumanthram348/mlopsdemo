# Base Image for ML App
FROM python:3

# Create app directory
WORKDIR /app

# Install app dependencies
COPY requirements.txt ./
#COPY model/rental_prediction_model.pkl ./
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD [ "python", "./app.py" ]
