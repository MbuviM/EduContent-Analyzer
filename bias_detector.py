import json
import os
from groq import Groq
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize the Groq client with your API key from environment variable
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Hardcoded dictionary of gender-biased words and their suggestions
GENDER_BIAS_DICTIONARY = {
    "mankind": ["humankind", "humanity", "people"],
    "manpower": ["workforce", "personnel", "staff", "human resources"],
    "chairman": ["chairperson", "chair", "head"],
    "businessman": ["business person", "business professional", "executive"],
    "businesswoman": ["business person", "business professional", "executive"],
    "salesman": ["salesperson", "sales representative", "sales associate"],
    "saleswoman": ["salesperson", "sales representative", "sales associate"],
    "policeman": ["police officer", "law enforcement officer"],
    "policewoman": ["police officer", "law enforcement officer"],
    "fireman": ["firefighter", "first responder"],
    "firewoman": ["firefighter", "first responder"],
    "mailman": ["mail carrier", "postal worker", "letter carrier"],
    "stewardess": ["flight attendant"],
    "steward": ["flight attendant"],
    "actress": ["actor", "performer"],
    "waitress": ["server", "wait staff", "waiter"],
    "waiter": ["server", "wait staff"],
    "housewife": ["homemaker", "stay-at-home parent"],
    "househusband": ["homemaker", "stay-at-home parent"],
    "manmade": ["artificial", "synthetic", "manufactured", "human-made"],
    "freshman": ["first-year student", "first-year"],
    "upperclassman": ["upper-year student", "upper-year"],
    "guys": ["everyone", "folks", "team", "all", "y'all"],
    "man-hours": ["work hours", "person-hours", "labor hours"],
    "man of the house": ["head of household", "family leader"],
    "woman of the house": ["head of household", "family leader"],
    "mother nature": ["nature", "the environment"],
    "king-size": ["extra-large", "oversized"],
    "queen-size": ["large"],
    "girl friday": ["assistant", "aide"],
    "gentleman's agreement": ["informal agreement", "unwritten agreement"],
    "master bedroom": ["primary bedroom", "main bedroom"],
    "mistress bedroom": ["primary bedroom", "main bedroom"],
    "maternal instinct": ["parental instinct"],
    "paternal instinct": ["parental instinct"],
    "career woman": ["professional", "career person"],
    "career man": ["professional", "career person"],
    "working mother": ["working parent"],
    "working father": ["working parent"],
    "forefathers": ["ancestors", "predecessors"],
    "lady doctor": ["doctor", "physician"],
    "male nurse": ["nurse"],
    "female lawyer": ["lawyer", "attorney"],
    "male secretary": ["secretary", "administrative assistant"],
    "tomboy": ["energetic child", "active child"],
    "sissy": ["timid person", "coward"],
    "grow a pair": ["be brave", "have courage"],
    "man up": ["be strong", "be resilient"],
    "like a girl": ["poorly", "weakly"],
    "boys will be boys": ["children will be children", "people will misbehave"],
    "don't be such a girl": ["don't be so sensitive", "be more resilient"],
    "old maid": ["unmarried person", "single person"],
    "spinster": ["unmarried person", "single person"],
    "bachelor": ["unmarried person", "single person"],
    "mothering": ["nurturing", "caring"],
    "fathering": ["parenting"],
    "workmanship": ["craftsmanship", "quality of work"],
    "mankind's achievements": ["human achievements", "humanity's achievements"],
    "the common man": ["the average person", "ordinary people"],
    "no man's land": ["unclaimed territory", "buffer zone"],
    "man-to-man defense": ["player-to-player defense", "one-on-one defense"],
    "man of the year": ["person of the year"],
    "woman of the year": ["person of the year"],
    "sportsmanship": ["fair play", "good sport behavior"],
    "maiden name": ["birth name", "family name"],
    "mankind's greatest adventure": ["humanity's greatest adventure"],
    "man of action": ["person of action", "go-getter"],
    "woman of action": ["person of action", "go-getter"]
}

def read_file_safely(file):
    encodings = ['utf-8', 'latin-1', 'utf-16', 'ISO-8859-1']
    for encoding in encodings:
        try:
            content = file.read()
            return content.decode(encoding)
        except UnicodeDecodeError:
            file.seek(0)  # Reset file pointer for next attempt
            continue
    raise UnicodeDecodeError(f"Failed to decode the file with tried encodings: {encodings}")

def analyze_text_with_dictionary(text):
    found_biases = []
    text_lower = text.lower()
    bias_score = 0
    
    for biased_word, alternatives in GENDER_BIAS_DICTIONARY.items():
        if biased_word.lower() in text_lower:
            found_biases.append({
                "biased_text": biased_word,
                "explanation": f"Found gender-specific term that could be replaced with more inclusive language.",
                "alternatives": alternatives
            })
            bias_score += 0.1
    
    return {
        "dictionary_results": {
            "bias_score": min(bias_score, 1.0),
            "suggestions": found_biases
        }
    }

def process_text_for_bias(text):
    # First, get dictionary results
    dictionary_analysis = analyze_text_with_dictionary(text)
    
    api_request_json = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": """You are an AI assistant specialized in detecting gender bias in text.
                Your task is to:
                1. Analyze the given text for gender bias, stereotypes, and unequal treatment
                2. Identify specific phrases or words that show gender bias
                3. Provide alternatives for biased language
                4. Assign a bias score from 0 (no bias) to 1 (extremely biased)
                
                Respond ONLY with a JSON object in this exact format:
                {
                    "bias_score": <number between 0 and 1>,
                    "suggestions": [
                        {
                            "biased_text": "<exact text from input that shows bias>",
                            "explanation": "<brief explanation of why this is biased>",
                            "alternatives": ["<alternative 1>", "<alternative 2>"]
                        }
                    ]
                }
                
                Focus ONLY on gender bias. Do not comment on JSON formatting or other types of bias."""
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,  # Lowered temperature for more consistent output
        "max_tokens": 1000,
        "response_format": {"type": "json_object"}
    }

    try:
        response = groq_client.chat.completions.create(**api_request_json)
        llm_analysis = json.loads(response.choices[0].message.content)
        
        # Combine dictionary and LLM results
        combined_suggestions = dictionary_analysis["dictionary_results"]["suggestions"]
        if "suggestions" in llm_analysis:
            combined_suggestions.extend(llm_analysis["suggestions"])
        
        combined_score = max(
            dictionary_analysis["dictionary_results"]["bias_score"],
            llm_analysis.get("bias_score", 0.0)
        )
        
        return {
            "bias_score": combined_score,
            "suggestions": combined_suggestions
        }
    except Exception as e:
        # Fallback to dictionary results if API fails
        return {
            "bias_score": dictionary_analysis["dictionary_results"]["bias_score"],
            "suggestions": dictionary_analysis["dictionary_results"]["suggestions"],
            "note": f"Used dictionary-only analysis due to API error: {str(e)}"
        }

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        text = read_file_safely(file)
        result = process_text_for_bias(text)
        return jsonify(result)
    except UnicodeDecodeError as e:
        return jsonify({'error': f"Failed to read file: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)