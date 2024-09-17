from transformers import pipeline

# Classification
classifier = pipeline("sentiment-analysis", model="microsoft/Phi-3-mini-128k-instruct", torch_dtype="auto", trust_remote_code=True)
result = classifier("I've been waiting for modela HuggingFace course my whole life.")
print("Classification: ", result)

# Multiple classification
classifier = pipeline("sentiment-analysis", model="microsoft/Phi-3-mini-128k-instruct", torch_dtype="auto", trust_remote_code=True)
result = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print("Multiple Classification: ", result)
classifier = None

# Zero-shot classification
# classifier = pipeline("zero-shot-classification", model="microsoft/Phi-3-mini-128k-instruct", torch_dtype="auto", trust_remote_code=True)
# result = classifier(
#     "This is a course about the Transformers library",
#     candidate_labels=["education", "politics", "business"]
# )
# print("Zero-Shot Classification: ", result)

# Text generation
generator = pipeline("text-generation", model="microsoft/Phi-3-mini-128k-instruct", torch_dtype="auto", trust_remote_code=True)
result = generator("In this course, we will teach you how to")
print("Text Generation: ", result)
generator = None

# Text generation with parameters
generator = pipeline("text-generation", model="microsoft/Phi-3-mini-128k-instruct", torch_dtype="auto", trust_remote_code=True)
result = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=1,
)
print("Text Generation With Parameters: ", result)
generator = None

# Mask filling
# unmasker = pipeline("fill-mask", model="microsoft/Phi-3-mini-128k-instruct", torch_dtype="auto", trust_remote_code=True)
# result = unmasker("This course will teach you all about <mask> models.", top_k=2)
# print("Mask Filling: ", result)
# unmasker = None

# Named entity recognition
# ner = pipeline("ner", grouped_entities=True, model="microsoft/Phi-3-mini-128k-instruct", torch_dtype="auto", trust_remote_code=True)
# result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
# print("Named Entity Recognition: ", result)
# ner = None

# Question answering
# question_answerer = pipeline("question-answering", model="microsoft/Phi-3-mini-128k-instruct", torch_dtype="auto", trust_remote_code=True)
# result = question_answerer(
#     question="Where do I work?",
#     context="My name is Sylvain and I work at Hugging Face in Brooklyn",
# )
# print("Question Answering: ", result)
# question_answerer = None

# Summarization
# summarizer = pipeline("summarization", model="microsoft/Phi-3-mini-128k-instruct", torch_dtype="auto", trust_remote_code=True)
# result = summarizer(
#     """
#     America has changed dramatically during recent years. Not only has the number of 
#     graduates in traditional engineering disciplines such as mechanical, civil, 
#     electrical, chemical, and aeronautical engineering declined, but in most of 
#     the premier American universities engineering curricula now concentrate on 
#     and encourage largely the study of engineering science. As a result, there 
#     are declining offerings in engineering subjects dealing with infrastructure, 
#     the environment, and related issues, and greater concentration on high 
#     technology subjects, largely supporting increasingly complex scientific 
#     developments. While the latter is important, it should not be at the expense 
#     of more traditional engineering.
# 
#     Rapidly developing economies such as China and India, as well as other 
#     industrial countries in Europe and Asia, continue to encourage and advance 
#     the teaching of engineering. Both China and India, respectively, graduate 
#     six and eight times as many traditional engineers as does the United States. 
#     Other industrial countries at minimum maintain their output, while America 
#     suffers an increasingly serious decline in the number of engineering graduates 
#     and a lack of well-educated engineers.
# """
# )
# print("Summarization: ", result)
# summarizer = None

# Translation
translator = pipeline("translation", model="microsoft/Phi-3-mini-128k-instruct", torch_dtype="auto", trust_remote_code=True)
result = translator("Ce cours est produit par Hugging Face.")
print("Translation: ", result)
translator = None
