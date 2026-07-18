# **Lazaridou, Peysakhovich & Baroni (2017) (Multi-Agent Cooperation and the Emergence of (Natural) Language)**

# Architecture Setup

1. Sender  - Sees target and distractor and sends one discrete symbol
2. Receiver - Sees 2 images in random order and receives the symbol
3. Reward - Score 1(if receiver guesses correctly) else Score 0
4. Goal - Two agents must invent a language from scratch

# Understanding the Basics

* VGG network turns picture into a checklist (embeddings) containing 1000 visual traits / dimensions (shininess, roundness etc.)
* The agents do not see the images, they only see the checklist.
* **Agnostic Sender (The Memorizer) :** Takes the checklists of the target and the distractor image and concatenates them together into one massive list and then it maps the combined vector to the vocabulary scores.
* **Informed Sender (The Comparator) :** It does not glues the two checklists unlike the agnostic sender, instead aligns them side-by-side. Uses 1D Convolutions to compare and sends the symbol representing the target more.
* **Example:** *Think of it like playing a game of Taboo. If your target is a "Zebra" and the distractor is a "Horse", the most useful word to send to your partner is "stripes". But if the distractor is a "Tiger", "stripes" isn't helpful anymore; you need to send a trait that separates them, like "hooves". The Informed Sender's architecture inherently allows it to make these contextual comparisons.*
* **Receiver :** Embeds the left image, right image and the incoming discrete symbol and then computes the dot product between the symbol vector and the image vector.
* **Communication Bottleneck :** The sender must send exactly one discrete symbol from its vocabulary and the weights are updated by the policy gradient rule via backpropagation.
* **Temperature :** It controls the balance between the exploration and exploitation.
* **Entropy Regularization:** Adds a penalty to the loss function if the agents get too predictable too quickly. Prevents early vocabulary collapse.
* **Semantic Purity:** measures how closely the AI's invented words align with real-world, human-defined categories (like animals, vehicles). Answers an important question : " Are the agents using specific symbols to represent specific concepts, or are they just using a chaotic, random code to pass the game? "

  * **How is it Calculated?**

    The researchers calculate semantic purity using a simple four-step clustering process:

    1. **Group by Word:** The system looks at all the images in the dataset and groups them into "boxes" based on the specific symbol the Sender network chooses most often for them.

    2. **Find the Winner:** The system looks inside each symbol's box and identifies the ground-truth category that appears the most. For example, if Box #5 contains 80% animals and 20% tools, "animals" is the majority category.

    3. **Compute the Percentage:** The purity score for that specific box becomes 80%. These scores are averaged across all used symbols to get the final Semantic Purity percentage.

    4. **Compare to Chance:** The final score is compared directly to a "Chance Baseline". This baseline represents the purity score you would get if you completely shuffled the words and assigned them to images at random. Anything significantly higher than chance proves the AI is developing a meaningful language.

# Hidden Training and Architecture Configurations

* **The Features Checklist (sm vs fc):** The paper tests two varieties of the VGG checklist: a 1000-dimensional list (`sm`) extracted from the final softmax layer, and a much deeper, detailed 4096-dimensional list (`fc`) taken from the second-to-last layer.

* **Vocabulary Sizes:** The authors explicitly limit and test their dictionary sizes using caps of either **10 symbols** or **100 symbols** to study how vocabulary restrictions change optimization.

* **Training Volume:** Agents are trained over a massive sequence of **50,000 games** processed in mini-batches of **32 games** at a time.

# The Core Biological Discovery

* **The "Living vs. Non-Living" Split:** When the Agnostic Sender collapses and uses only 2 symbols, its choices are not random. It naturally learns to partition the visual universe into "living things" vs. "non-living things". This is incredibly significant because cognitive scientists recognize this exact contrast as the absolute core, primary building block of the human brain's semantic memory.

# Advanced Environmental Tweaks

* **Object-Level Reference (Section 4.1):** To stop the agents from "cheating" by matching exact visual quirks or pixels, the authors strip out "common knowledge" from the game environment. They do this by showing the Sender one image of a concept (e.g., a Chihuahua) and showing the Receiver a completely different image of that same concept (e.g., a Boston Terrier). This forces the neural networks to coordinate entirely on abstract category definitions.

* **Language Grounding (Section 5):** Taking explicit inspiration from AlphaGo's multi-stage learning, the Sender splits its training time 50/50 between playing the communication game and completing a supervised image-labeling task. Because both tasks share the exact same visual embedding matrix, the arbitrary symbols are mathematically anchored into conventional human words.

# Real-World Human Testing

* **The CrowdFlower Online Experiment:** To see if the AI language was portable, the researchers replaced the AI Receiver with real human players on a crowdsourcing platform. Even when the AI was forced to talk about brand-new object scenes it had never seen before, humans successfully picked the correct target image **68% of the time** based entirely on the word emitted by the Sender.

* **Poetic Language Flexibility (Metonymic Links):** When the AI was faced with a new scene and didn't possess the exact word for it, it displayed human-like semantic shifts. For instance, it emitted the word *"dolphin"* to signal a picture of an empty blue stretch of sea, and the word *"fence"* to point out a patch of dry dirt. Because humans possess the cognitive flexibility to interpret these poetic, contextual context-clues, they were able to play the game with the AI successfully.
