# src/ai_modules/scene_detection.py

import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip
from datetime import datetime
import json

class SceneDetector:
    def __init__(self, threshold=30.0, min_scene_length=2.0):
        """
        Initialize Scene Detector
        
        Args:
            threshold (float): Sensitivity for scene change detection (lower = more sensitive)
            min_scene_length (float): Minimum scene length in seconds
        """
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.scene_timestamps = []
        
    def calculate_frame_difference(self, frame1, frame2):
        """Calculate the difference between two frames"""
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram difference
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Compare histograms using correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Convert correlation to difference score
        difference = (1 - correlation) * 100
        
        return difference
    
    def detect_scenes(self, video_path, progress_callback=None):
        """
        Detect scene changes in video
        
        Args:
            video_path (str): Path to input video
            progress_callback (function): Optional callback for progress updates
            
        Returns:
            list: List of scene timestamps [(start, end), ...]
        """
        print(f"Analyzing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Error: Could not open video {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        # Initialize variables
        scene_changes = [0]  # Start with first frame
        prev_frame = None
        frame_count = 0
        
        # Process video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if prev_frame is not None:
                # Calculate difference between current and previous frame
                difference = self.calculate_frame_difference(prev_frame, frame)
                
                # Check if difference exceeds threshold
                if difference > self.threshold:
                    timestamp = frame_count / fps
                    # Ensure minimum scene length
                    if len(scene_changes) == 1 or (timestamp - scene_changes[-1]) >= self.min_scene_length:
                        scene_changes.append(timestamp)
                        print(f"Scene change detected at {timestamp:.2f}s (difference: {difference:.2f})")
            
            prev_frame = frame.copy()
            frame_count += 1
            
            # Progress callback
            if progress_callback and frame_count % 30 == 0:  # Update every 30 frames
                progress = (frame_count / total_frames) * 100
                progress_callback(progress)
        
        # Add end timestamp
        scene_changes.append(duration)
        
        # Create scene pairs (start, end)
        self.scene_timestamps = [(scene_changes[i], scene_changes[i + 1]) 
                                for i in range(len(scene_changes) - 1)]
        
        cap.release()
        
        print(f"Detected {len(self.scene_timestamps)} scenes")
        return self.scene_timestamps
    
    def split_video_into_scenes(self, video_path, output_dir, scene_timestamps=None):
        """
        Split video into scene clips
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save scene clips
            scene_timestamps (list): Optional list of scene timestamps
            
        Returns:
            list: List of created clip file paths
        """
        if scene_timestamps is None:
            scene_timestamps = self.scene_timestamps
        
        if not scene_timestamps:
            raise Exception("No scene timestamps available. Run detect_scenes() first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video filename without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Load video with moviepy
        print("Loading video for splitting...")
        video = VideoFileClip(video_path)
        
        created_clips = []
        
        # Split video into scenes
        for i, (start_time, end_time) in enumerate(scene_timestamps):
            scene_duration = end_time - start_time
            
            if scene_duration < 0.5:  # Skip very short scenes
                print(f"Skipping scene {i+1} (too short: {scene_duration:.2f}s)")
                continue
            
            print(f"Creating scene {i+1}/{len(scene_timestamps)}: "
                  f"{start_time:.2f}s - {end_time:.2f}s ({scene_duration:.2f}s)")
            
            try:
                # Extract scene clip
                scene_clip = video.subclip(start_time, end_time)
                
                # Generate output filename
                output_filename = f"{video_name}_scene_{i+1:03d}_{start_time:.1f}s-{end_time:.1f}s.mp4"
                output_path = os.path.join(output_dir, output_filename)
                
                # Write scene clip
                scene_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None
                )
                
                scene_clip.close()
                created_clips.append(output_path)
                
                print(f"✓ Saved: {output_filename}")
                
            except Exception as e:
                print(f"✗ Error creating scene {i+1}: {str(e)}")
                continue
        
        video.close()
        
        # Save scene information to JSON
        scene_info = {
            'source_video': video_path,
            'total_scenes': len(scene_timestamps),
            'created_clips': len(created_clips),
            'detection_threshold': self.threshold,
            'min_scene_length': self.min_scene_length,
            'scenes': [
                {
                    'scene_number': i + 1,
                    'start_time': start,
                    'end_time': end,
                    'duration': end - start
                }
                for i, (start, end) in enumerate(scene_timestamps)
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        info_file = os.path.join(output_dir, f"{video_name}_scene_info.json")
        with open(info_file, 'w') as f:
            json.dump(scene_info, f, indent=2)
        
        print(f"\n✓ Scene detection complete!")
        print(f"  - Created {len(created_clips)} scene clips")
        print(f"  - Saved to: {output_dir}")
        print(f"  - Scene info saved to: {info_file}")
        
        return created_clips
    
    def process_video(self, video_path, output_dir=None, progress_callback=None):
        """
        Complete pipeline: detect scenes and split video
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Output directory (auto-generated if None)
            progress_callback (function): Progress callback function
            
        Returns:
            tuple: (scene_timestamps, created_clip_paths)
        """
        # Auto-generate output directory if not provided
        if output_dir is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"output/scenes_{video_name}_{timestamp}"
        
        # Detect scenes
        print("Step 1: Detecting scenes...")
        scene_timestamps = self.detect_scenes(video_path, progress_callback)
        
        # Split video
        print("\nStep 2: Splitting video into scenes...")
        created_clips = self.split_video_into_scenes(video_path, output_dir, scene_timestamps)
        
        return scene_timestamps, created_clips


# Example usage function
def demo_scene_detection():
    """Demo function showing how to use the SceneDetector"""
    
    # Initialize detector
    detector = SceneDetector(
        threshold=25.0,      # Adjust sensitivity (lower = more sensitive)
        min_scene_length=3.0 # Minimum scene length in seconds
    )
    
    # Progress callback function
    def show_progress(percentage):
        print(f"Progress: {percentage:.1f}%")
    
    # Process video
    video_path = "data/sample_videos/input.mp4"  # Change to your video path
    output_dir = "data/output/scenes"
    
    try:
        scene_timestamps, created_clips = detector.process_video(
            video_path=video_path,
            output_dir=output_dir,
            progress_callback=show_progress
        )
        
        print("\n" + "="*50)
        print("SCENE DETECTION SUMMARY")
        print("="*50)
        for i, (start, end) in enumerate(scene_timestamps):
            print(f"Scene {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    demo_scene_detection()