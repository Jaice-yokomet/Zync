# src/main.py

import sys
import os
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from ai_modules.scene_detection import SceneDetector

def main():
    """Main application entry point"""
    print("="*60)
    print("AI VIDEO EDITOR - Scene Detection Module")
    print("="*60)
    
    # Configuration
    input_video = "data/sample_videos/input.mp4"  # Change this to your video path
    output_base_dir = "data/output"
    
    # Scene detection parameters
    detection_threshold = 25.0  # Lower = more sensitive to scene changes
    min_scene_length = 2.0      # Minimum scene length in seconds
    
    print(f"Input video: {input_video}")
    print(f"Output directory: {output_base_dir}")
    print(f"Detection threshold: {detection_threshold}")
    print(f"Minimum scene length: {min_scene_length}s")
    print("-" * 60)
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"❌ Error: Video file not found: {input_video}")
        print("\nPlease:")
        print("1. Create the directory: data/sample_videos/")
        print("2. Place your video file there and rename it to 'input.mp4'")
        print("3. Or update the 'input_video' path in main.py")
        return
    
    try:
        # Initialize scene detector
        detector = SceneDetector(
            threshold=detection_threshold,
            min_scene_length=min_scene_length
        )
        
        # Progress callback
        def show_progress(percentage):
            if int(percentage) % 10 == 0:  # Show progress every 10%
                print(f"🔄 Analysis progress: {percentage:.1f}%")
        
        # Generate output directory with timestamp
        video_name = os.path.splitext(os.path.basename(input_video))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base_dir, f"scenes_{video_name}_{timestamp}")
        
        print("🎬 Starting scene detection and video splitting...")
        print()
        
        # Process the video
        scene_timestamps, created_clips = detector.process_video(
            video_path=input_video,
            output_dir=output_dir,
            progress_callback=show_progress
        )
        
        # Display results
        print("\n" + "="*60)
        print("📊 PROCESSING COMPLETE!")
        print("="*60)
        print(f"✅ Total scenes detected: {len(scene_timestamps)}")
        print(f"✅ Scene clips created: {len(created_clips)}")
        print(f"✅ Output location: {output_dir}")
        
        print("\n📋 Scene Breakdown:")
        print("-" * 40)
        total_duration = 0
        for i, (start, end) in enumerate(scene_timestamps):
            duration = end - start
            total_duration += duration
            print(f"Scene {i+1:2d}: {start:6.1f}s - {end:6.1f}s ({duration:5.1f}s)")
        
        print("-" * 40)
        print(f"Total duration: {total_duration:.1f}s")
        
        print(f"\n📁 Files created:")
        for clip_path in created_clips:
            filename = os.path.basename(clip_path)
            file_size = os.path.getsize(clip_path) / (1024*1024)  # Convert to MB
            print(f"  • {filename} ({file_size:.1f} MB)")
        
        print(f"\n💡 Scene info saved to: {video_name}_scene_info.json")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the video file is not corrupted")
        print("2. Check if you have enough disk space")
        print("3. Ensure video format is supported (mp4, avi, mov, etc.)")
        print("4. Try reducing the detection threshold if no scenes are detected")


def interactive_mode():
    """Interactive mode for custom video processing"""
    print("\n🎯 INTERACTIVE MODE")
    print("-" * 30)
    
    # Get video path from user
    video_path = input("Enter video file path: ").strip()
    
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return
    
    # Get parameters from user
    try:
        threshold = float(input("Detection threshold (default 25.0): ") or "25.0")
        min_length = float(input("Minimum scene length in seconds (default 2.0): ") or "2.0")
    except ValueError:
        print("❌ Invalid input. Using default values.")
        threshold = 25.0
        min_length = 2.0
    
    # Get output directory
    output_dir = input("Output directory (press Enter for auto): ").strip()
    if not output_dir:
        output_dir = None
    
    # Process video
    detector = SceneDetector(threshold=threshold, min_scene_length=min_length)
    
    def progress_callback(pct):
        print(f"Progress: {pct:.1f}%")
    
    try:
        scene_timestamps, created_clips = detector.process_video(
            video_path, output_dir, progress_callback
        )
        print(f"\n✅ Success! Created {len(created_clips)} scene clips.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
