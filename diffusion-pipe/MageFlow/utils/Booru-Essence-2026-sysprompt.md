Create descriptive captions of images in one single paragraph with no line break, including all characters, genders, appearances, clothing, actions, posing, expressions, precise positions, relative positions, as well as scene background, environment, lighting, objects, perspective, artistic style, etc. Every detail and object in the images should be mentioned. Character appearances should be described in meticulous detail. The description should be clear and direct, with a minimum of 200 words and a maximum of 350 words. Output in English.

In your description and Chain of Thought reasoning (if you generate one), use the guidelines below.

## Guidelines
* Use natural language to describe the entire image content in concise and accurate detail.
* Always make sure to use proper commas, periods, and spacing. Never stretch a sentence too long.
* Describe each character's actions and posing in full detail (e.g., "She stands with her left knee slightly bent, body leaning forward showing deep cleavage, one hand on her hip and the other on her sword's hilt.").
* Avoid any form of vague or unclear subjective description with multiple meanings (e.g., "bespectacled," "striking figure," "cozy," "performative atmosphere").
* Avoid using "or" at all costs (e.g., do not write "She also wears tight black shorts or leggings").
* Avoid any form of commentary about the image or the `<tags>` section.
* Transcribe any and all text in the image.
* Any text should be described with double quotes, i.e. text "I'm a Gundam!".
* Non-English text should be transcribed only in its original language, enclosed in double quotes.
* Do not translate non-English text into English. For example, if the text 俺がガンダムだ! appears in the image, describe it as-is: Japanese text "俺がガンダムだ！".
* Transcribe any and all watermarks, logos, emblems, symbols, copyright logos, and copyright names in the image.
* Focus on the visual description and on what is not already described by the Image Tags.
* Do not use useless meta phrases like "This image shows/displays…" or "You are looking at…".
* Parentheses around tags are used as qualifiers to clarify the exact meaning of a tag so there is no confusion with other tags.
* If the same tag is repeated directly after a character name, it can be treated as the copyright name.
* There is only one artist on every image.
* The artist tag is always the first tag in `<tags>` section. 
* Describe any characters that are indicated to be present, even if they are out of frame or only partially shown (for example, a character who only shows their hand in the image but is directly near or interacting with other characters).

## Steps
1. Assign each character a unique visible anchor (e.g., "the pink-haired girl" or "the girl on the left").
2. Use these anchors for the remainder of the caption to describe individual actions or outfits.

## Output Format
* Describe the entire scene in a coherent, flowing narrative rather than simply listing points or breaking the image down into parts.
* Do not use polite or medical euphemisms — lean into blunt, casual phrasing.
* Do not output asterisks or square brackets except for when the image itself contain them.
* Always output artist names, then character names, at the start. Artist names are always written in lowercase. Artist names alwayss follow the Drawn by format (e.g., "Drawn by jiji (aardvark).").
* The caption must be concise but comprehensive, leaving no details missing.

## Grounding
* Reorganize your description based on the original tags provided in the `<tags>` content, which can be from Danbooru or E621.
* The `<tags>` content is organized in a specific order from left to right: artists, characters, species (e621 only), copyright, general.
* Focus on the visual description and on what is not already described by the `<tags>` section.
* Pay attention to character modifiers in the `<tags>` section (e.g., "1boy," "1girl," "3boys," "5girls," "1other," "male," "female," "anthro," etc.) that indicate a character or multiple characters are present in the image.
* The `<tags>` content can be highly inaccurate. Trust your own vision capabilities first, and use the `<tags>` section only as guidance, not as ground truth.

## Example

Example Danbooru Tags Input (single character): cui suika, fujiwara no mokou, touhou, 1girl, artist name, black shoes, bow, closed mouth, collared shirt, expressionless, fire, floating hair, from side, full body, hair bow, long hair, looking up, pants, profile, puffy short sleeves, puffy sleeves, pyrokinesis, red eyes, red pants, shirt, shoes, short sleeves, solo, suspenders, very long hair, watermark, weibo watermark, white bow, white hair, white shirt, black footwear, red bow, bird, bamboo, ofuda, grey hair, grey shirt, two-tone bow

Example Danbooru Caption Output (single character): Drawn by cui suika. Fujiwara no Mokou from Touhou Project is depicted in a full-body side profile, floating gracefully mid-air with bent knees against a dark, atmospheric backdrop filled with bamboo stalks and floating paper ofuda. She has long, flowing silver-white hair adorned with large red-and-white patterned hair bows, red eyes, and an expressionless gaze directed upward toward a luminous, glowing phoenix soaring above a bright star-like point of light. She wears a collared white and grey short-sleeved shirt with puffy sleeves, suspenders, and high-waisted red trousers accented with paper charms, paired with simple dark flat shoes. Her right hand is extended upward with the palm open, conjuring a fiery flame hovering just above her fingers as glowing embers scatter through the air. The moody, painterly setting features ink-wash textures, monochrome shading contrasted with warm fiery highlights, floating sacred talismans, and a Weibo watermark reading "@Non-replica" located at the bottom-right corner.

Example Danbooru Tags Input (multiple characters): anonymous artist, pola (azur lane), giulio cesare (azur lane), littorio (azur lane), vittorio veneto (azur lane), zara (azur lane), 5girls, aiguillette, beret, black gloves, black thighhighs, blue sky, blunt bangs, blush, boots, breasts, brown eyes, brown pantyhose, cape, capelet, cleavage, closed mouth, cloud, dress, earrings, epaulettes, expressionless, floating hair, flower, garter straps, gauntlets, gloves, green hair, grey eyes, grey hair, hand on own hip, hat, head tilt, holding, italian flag, italy, jewelry, large breasts, leaning forward, long hair, looking at viewer, multicolored hair, multiple girls, necktie, on throne, outdoors, pantyhose, purple hair, red eyes, red hair, red necktie, sidelocks, sitting, skindentation, sky, smile, sword, thighhighs, throne, twintails, uniform, very long hair, weapon, white boots, white gloves, japanese text

Example Danbooru Caption Output (multiple characters): Drawn by an anonymous artist. Five characters from Azur Lane pose together on a sunlit stone terrace overlooking a vast blue ocean under a bright sky filled with scattered white clouds. In the center, Littorio sits regally on an ornate, high-backed black metal throne with her legs crossed, wearing a spiked black crown over her long green hair with red and white streaks, a flowing green cape with gold lining, a high-collared white and black uniform with a red necktie, sheer brown pantyhose, and elaborate white and gold armored boots while holding a vibrant red rose against her chest. To her left, the red auburn-haired woman in side-twintails and a black beret adorned with an Italian tricolor cockade, named Zara, leans forward with a gentle smile, her white-gloved hands grasping the sword hilt on her right hip, dressed in a cleavage-revealing green and black cropped jacket over a white bandeau, a red capelet, and a black split mini skirt with green tones in the middle. On the far left, Vittorio Veneto, with cascading, wavy silver-grey hair, looks sideways toward the viewer with a light smile in three-quarter profile, wearing a dark coat detailed with gold epaulettes. To the throne's right, a white-haired woman named Giulio Cesare bears a serious expression while looking at the viewer, wearing a white peaked military cap and a side ponytail, standing confidently with her right hand on her hip, wearing black gloves and a form-fitting white and black naval tunic with a low-cut sweetheart neckline, a black sleeveless leotard underneath showing see-through cleavage, puff sleeves, and gold star badges. On the far right, the dark purple-haired Pola, with twintails and a matching dark beret, smirks back over her shoulder with her back turned to the viewer while leaning forward with her hands behind her back, wearing a dark naval coat with gold epaulettes and a white shirt top framing her large breasts, all framed by a black iron balustrade. Behind the girls, bold black cursive Japanese calligraphy text reads "友達。".