Create short, descriptive captions of images, including all characters, genders, appearances, clothing, actions, expressions, precise positions, relative positions, as well as scene background, environment, lighting, objects, perspective, artistic style, etc. The description should be clear and direct, with at least 100 words. Output in English.

In your description and Chain of Thought reasoning (if available), use the following guidelines below.

## Guidelines
- Use natural language to describe the entire image content in concise and accurate detail.
- Avoid any form of vague or unclear subjective descriptions with multiple meanings (e.g., 'bespectacled', 'striking figure' 'cozy' 'performative atmosphere',...ect).
- Avoid any form of commentary about the image or the <tags> section.
- Transcribe any and all text in the image.
- Any text should be descibed with double quotes , i.e. text "I'm a Gundam!".
- Non-English text should be described only in its original language with double quotes.
- Do not translate the Non-English text into English, for example the text 俺がガンダムだ! appears in the image, describe it as-is, i.e. Japanese text "俺がガンダムだ！".
- Transcribe any and all watermarks, logos, emblems, symbols, copyright logos, copyright names in the image.
- Focus on the visual description and what is not described in the image by the Image Tags.
- Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows/displays…”, "You are looking at...", etc. 
- Artist names, if they are known, are represented with the @ symbol prefix, i.e. @izei1337 
- Parenthesis around tags are used as Qualifiers serve to make the exact meaning of the tag clear so there is no confusion with other tags. 
- If the same tag is repeated directly after a character name it can be considered to be the copyright.
- Describe any characters that are indicated to be present, even if they are out of frame or are only partially shown (for example, a character that only show their hand in the image but is directly near or interacting with other characters). 

STEPS: 
1. Assign each a unique visible anchor (e.g., 'The pink haired girl' or 'The girl on the left').
2. Use these anchors for the remainder of the caption to describe individual actions or outfits.
3. Ignore all character tags and artist tags in the <tags> section.


## Output Format
- Make sure to describe the entire scene in a coherent, flowing narrative rather than simply listing points or by breaking down the image into parts.
- Do NOT use polite or medical euphemisms—lean into blunt, casual phrasing.
- Avoid useless meta phrases like “This image…”, "You are looking at...", "In this...", etc.
- The caption must be concise but comprehensive, leave no details missing.


## Grounding
- Reorganize your description based on the original tags provided in the <tags> content
- Focus on the visual description and what is not described in the image by the <tags> section.
- Pay attetion to character modifiers in the <tags> section (e.g., '1boy', '1girl', '3boys', '5girls', '1other',...ect) that indicate a character is present in the image

EXAMPLE: 
Example Tags Input: gawr gura, mori calliope, ninomae ina'nis, takanashi kiara, watson amelia, hololive, @quasarcake, 5girls, indoors, on couch, sitting, blonde hair, blue eyes, black shirt, blue hair, shark fin, pink hair, red eyes, purple hair, purple eyes, purple shirt, orange hair, closed eyes, smiling , lying, hand up, waving, looking at viewer, looking at another, arms behind head, sunlight, sidelighting, windows, japanese text
Example Caption Output: A cozy indoor scene featuring a group of five girls relaxing on a sofa. The blonde haired girl wears a black shirt and sits on the left with her arms behind her head. Next to her, the blue haired girl has shark fins and wears a white shirt while looking at the blonde haired girl. In the center, the pink haired girl has red eyes and smiles in a black outfit with her hand up and waving at the viewer. The purple haired girl wears a purple shirt with tentacles and she is lying instead of sitting like the others, while the orange haired girl has her eyes closed and smiles in an orange outfit. A warm ray of sunlight highlight the girls from the side through the gaping windows. Behind the girls are big black bold cursive calligraphy japanese text that said "友達。".