import csv
import os
from datetime import datetime
from typing import List, Dict

class AttendanceTracker:
    def __init__(self, csv_path: str = "../logs/attendance_log.csv"):
        """
        Initialize the AttendanceTracker for recording unique class names.

        Args:
            csv_path (str): Path to the CSV file for storing attendance logs.
        """
        self.csv_path = csv_path
        self.tracked_classes: set = set()  # Store unique class names that have been logged
        self.attendance_log: List[Dict] = []
        self._initialize_csv()
        self._load_existing_attendance()

    def _initialize_csv(self):
        """
        Initialize the CSV file with headers if it doesn't exist.
        """
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['name', 'timestamp'])
                writer.writeheader()

    def _load_existing_attendance(self):
        """
        Load existing attendance records from the CSV file.
        """
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            with open(self.csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                self.attendance_log = [row for row in reader]
                # Populate tracked_classes with existing class names
                self.tracked_classes = {row['name'] for row in self.attendance_log}

    def track_objects(self, detections: List[Dict], session_id: str) -> List[Dict]:
        """
        Track detected objects and log each unique class name only once.

        Args:
            detections (List[Dict]): List of detection dictionaries containing class_name and box.
            session_id (str): Session ID for the client.

        Returns:
            List[Dict]: Current attendance log.
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for detection in detections:
            # class_name = detection['class_name']
            class_name = detection.get("label") or detection.get("class_name", "unknown")
            # Only log the class if it hasn't been logged before
            if class_name not in self.tracked_classes:
                self._log_attendance(class_name, current_time)
                self.tracked_classes.add(class_name)

        return self.attendance_log

    def _log_attendance(self, name: str, timestamp: str):
        """
        Log a new attendance record to the list and CSV file.

        Args:
            name (str): Name of the detected class.
            timestamp (str): Timestamp of the detection.
        """
        record = {'name': name, 'timestamp': timestamp}
        self.attendance_log.append(record)
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['name', 'timestamp'])
            writer.writerow(record)

    def get_attendance_log(self) -> List[Dict]:
        """
        Get the current attendance log.

        Returns:
            List[Dict]: List of attendance records.
        """
        return self.attendance_log

    def export_to_csv(self) -> str:
        """
        Export the attendance log to the CSV file.

        Returns:
            str: Path to the CSV file.
        """
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['name', 'timestamp'])
            writer.writeheader()
            writer.writerows(self.attendance_log)
        return self.csv_path