import os
import argparse
import json
from dotenv import load_dotenv
import openai
import moviepy.editor as mp
from pydub import AudioSegment
from pydub.utils import make_chunks

# Load environment variables
load_dotenv()


# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_KEY")

# Function to print logs based on feature flag
def log(message, log_flag):
    if log_flag:
        print(message)

# Function to extract audio from video
def extract_audio(video_path, log_flag):
    log("Extracting audio from video...", log_flag)
    video = mp.VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".mp3")
    video.audio.write_audiofile(audio_path)
    log(f"Audio extracted to {audio_path}", log_flag)
    return audio_path

# Function to split audio into chunks
def split_audio(audio_path, chunk_length_ms=600000, log_flag=False):
    log("Splitting audio into chunks...", log_flag)
    audio = AudioSegment.from_file(audio_path, format="mp3")
    chunks = make_chunks(audio, chunk_length_ms)
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_name = f"{audio_path[:-4]}_chunk{i}.mp3"
        chunk.export(chunk_name, format="mp3")
        chunk_files.append(chunk_name)
        log(f"Chunk {i} saved as {chunk_name}", log_flag)
    log("Audio splitting completed.", log_flag)
    return chunk_files

# Function to transcribe audio with context
def transcribe_audio_with_context(audio_path, previous_text, log_flag):
    log(f"Transcribing audio {audio_path} with context...", log_flag)
    with open(audio_path, "rb") as audio_file:
        prompt_text = previous_text[-224:] if previous_text else ""
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            language="en",
            file=audio_file,
            prompt=prompt_text,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    log(f"Transcription completed for {audio_path}.", log_flag)
    return response

# Function to summarize text
def summarize_text(transcription, text, log_flag = False):
    log("Summarizing transcription...", log_flag)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "system", "content": f"RAW TRANSCRIPTION:\n\n{transcription}"},
            {"role": "user", "content": f"Determine the type of the meeting content, summarize the following content and list the main key points discussed in the meeting content:\n\n{text}"}
        ],
        temperature=0.0,
    )
    summary = response.choices[0].message.content
    log("Summary completed.", log_flag)
    return summary

# Main function
def main(video_path, log_flag):
    # Extract audio from video
    audio_path = extract_audio(video_path, log_flag)

    # Split audio into chunks
    chunk_files = split_audio(audio_path, log_flag=log_flag)

    # Transcribe each chunk with context and save the transcriptions
    full_context = ""
    previous_text = ""
    for i, chunk_file in enumerate(chunk_files):
        transcription = transcribe_audio_with_context(chunk_file, previous_text, log_flag)
        log(f"Transcription for {chunk_file}:\n{transcription}", log_flag)

        transcription_dict = transcription.to_dict() if hasattr(transcription, 'to_dict') else transcription
        
        with open(f"{video_path[:-4]}_chunk{i}_transcription.json", "w") as f:
            json.dump(transcription_dict, f, indent=4)
        
        previous_text = transcription_dict['text']
        full_context += previous_text

    summary = summarize_text(full_context, log_flag)

    # Save summary to file
    with open(f"{video_path[:-4]}_summary.txt", "w") as f:
        f.write(summary)

    log("Transcriptions and summary have been saved to files.", log_flag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe and summarize video/audio files.')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--log', action='store_true', help='Enable logging')

    args = parser.parse_args()

    main(args.video_path, args.log)
