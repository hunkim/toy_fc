# Streamlit App Setup

This guide will help you set up and run the Streamlit app.

## Configuration

Before running the app, you need to configure some settings. Follow these steps:

1. Create a file named `.streamlit/secrets.toml` in your project directory if it doesn't already exist.

2. Add the following content to the `.streamlit/secrets.toml` file:

   ```toml
   UPSTAGE_API_KEY = "up_xxxxx"
   MODEL_NAME = "solar-pro"
   ```

   Replace `"up_xxxxx"` with your actual Upstage API key.

## Running the App

Once you've completed the configuration, you can run the app using the following command:

```bash
make app
```# solar_monitor
# toy_fc
