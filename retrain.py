from train import train_model
import datetime

if __name__ == "__main__":
    print(f"🔁 Starting retraining job at {datetime.datetime.now()}")
    train_model()
    print("✅ Retraining complete!")
