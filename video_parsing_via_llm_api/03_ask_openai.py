import json
import openai
import base64


# Function to encode the image
def encode_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_file_with_openai(file_path: str, prompt: str) -> dict:
    """
    Sends a local file and a prompt to OpenAI for analysis.

    Parameters:
        file_path (str): The path to the local file to analyze.
        prompt (str): The prompt or instructions for the analysis.

    Returns:
        dict: The response from the OpenAI API.
    """
    try:
        base64_image = encode_image(file_path)

        file_response = openai.files.create(file=open(file_path, 'rb'), purpose='vision')

        # Create the analysis request
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI which is presented with a screenshot from a CS2-stream. The screenshot might either show a scene from the gameplay or a waiting screen."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                ]}
            ],
        )

        return response

    except Exception as e:
        return {"error": str(e)}


# Todo: Engineer a better prompt. Maybe break it down to multiple separate questions, as that seemed to have worked better.
PROMPT = """
First, decide whether the following screenshot shows gameplay or not.

If it is a gameplay screenshot, analyse the screenshot and determine the teams playing, the time in the round, the terrorist round score, the counterterrorist round score, 
whether the bomb is planted or not, the number of terrorist players alive, the number of counterterrorist players alive, a list of weapons of the terrorist players, 
a list of weapons of the counterterrorist players. If available, parse the terrorist equipment value and the counterterrorist equipment value.
Present the format in the following format (the definition is given as Pydantic object, but you may choose JSON as output; where it is a list, always put the terrorist team first):
class snapshot(pydantic.BaseModel):
    round_time: datetime
    terrorist_team: str
    counterterrorist_team: str
    terrorist_score: int
    counterterrorist_score: int
    bomb_is_planted: bool
    num_terrorist_alive: int
    num_counterterrorist_alive: int
    terrorist_weapons: list(str)
    counterterrorist_weapons: list(str)
    terrorist_equipment_value: Optional[int]
    counterterrorist_equipment_value: Optional[int]

If the round time in the screenshot is red, then give the round time as negative value.

If it is a screenshot showing a waiting screen, simply return the string "waiting screen".
"""


if __name__ == '__main__':
    # Run through 10 local files:
    # Todo: Generate the file list from local directory.
    for k in range(11):
        filename = f'test_segment_{str(k).zfill(4)}.ts.jpg'
        response = analyze_file_with_openai(f'video_parsing_via_llm_api/{filename}', PROMPT)
        content = response.choices[0].message.content
        if len(content.splitlines()) > 2:
            content = ''.join(content.splitlines()[1:-1])
        else:
            content = '{}'

        json.dump(content, open(f'video_parsing_via_llm_api/{str(k).zfill(4)}_data.json', 'wt'))

        print(f'\nProcessed file {k} with a content of {content}')
