from typing import Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import tool_example_to_messages
from tools.pubmed import fetch_pubmed_title_abstract
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=".env.local")

llm = ChatOpenAI(model="gpt-4o-mini")

class Article(BaseModel):
    """Information about an article."""
    title: Optional[str] = Field(default=None, description="The title of the article")
    abstract: Optional[str] = Field(default=None, description="Generate a 3-5 sentence summary of the article")
    domain: Optional[str] = Field(default=None, description="The domain of the article (ex: physics, biology, mathematics, finance.)")
    topic: Optional[str] = Field(default=None, description="The main topic of the article (ex: quantum mechanics, cell biology, calculus, stock market.)")
    subtopic: Optional[str] = Field(default=None, description="The subtopic of the article if any (ex: string theory, cell division, integration, options trading.)")

class Data(BaseModel):
    """Extracted data about articles."""
    articles: List[Article]

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)


structured_llm = llm.with_structured_output(schema=Data)

examples = [
    (
        "This study explored the relationship between the foot arch stiffness and windlass mechanism, focusing on the contribution of the posterior transverse arch. Understanding the changing characteristics of foot stiffness is critical for providing a scientific basis for treating foot-related diseases. Based on a healthy male's computed tomography, kinematic, and dynamics data, a foot musculoskeletal finite element model with a dorsiflexion angle of 30°of metatarsophalangeal joint was established. Analyze the changes in stress distribution of the plantar fascia, metatarsophalangeal joint angle, arch height, and length during barefoot walking as the stiffness of the plantar fascia varies from 25 % to 200 %. For validation, the simulated arch parameters were compared with the dual fluorescence imaging system measurements. The width of transverse arch, height, and length of longitudinal arch measured by the dual fluorescence imaging system were 45.14 ± 1.63 mm, 29.29 ± 1.57 mm, and 155.16 ± 2.69 mm, respectively. The results of the simulation were 46.51 mm, 29.96 mm, and 156.71 mm, respectively. With the increase of plantar fascia stiffness, the effect of the windlass mechanism increased, the flexion angle of the metatarsophalangeal joint decreased, the distal stress of plantar fascia decreased gradually, while the proximal and middle stress increased, the transverse arch angle increased, but when the plantar fascia stiffness exceeds 150 %, the transverse arch angle decreases. The increase of plantar fascia stiffness will increase the effect of the windlass mechanism but decrease the flexion angle of the metatarsophalangeal joint. The stiffness of the plantar fascia influences the behavior of the plantar fascia. The plantar fascia stiffness affects the distal tension of the plantar fascia by affecting the flexion of the metatarsophalangeal joint in the plantar windlass mechanism. It affects the stiffness of the transverse arch of the foot together with the ground reaction force acting on the distal metatarsal.",
        Data(articles=[Article(title="Foot Arch Stiffness and Windlass Mechanism", abstract=None, domain="Biomechanics", topic="Foot Arch Stiffness", subtopic="Windlass Mechanism")]),
    ),
    (
        "Accuracy of genomic prediction depends on the heritability of the trait, the size of the training set, the relationship of the candidates to the training set, and the <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mtext>Min</mtext> <mo>(</mo> <msub><mi>N</mi> <mtext>QTL</mtext></msub> <mo>,</mo> <msub><mi>M</mi> <mi>e</mi></msub> <mo>)</mo></mrow> </math> , where <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><msub><mi>N</mi> <mtext>QTL</mtext></msub> </math> is the number of QTL and <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><msub><mi>M</mi> <mi>e</mi></msub> </math> is the number of independently segregating chromosomal segments. Due to LD, the number <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><msub><mi>Q</mi> <mi>e</mi></msub> </math> of independently segregating QTL (effective QTL) can be lower than <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mtext>Min</mtext> <mo>(</mo> <msub><mi>N</mi> <mtext>QTL</mtext></msub> <mo>,</mo> <msub><mi>M</mi> <mi>e</mi></msub> <mo>)</mo></mrow> </math> . In this paper, we show that <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><msub><mi>Q</mi> <mi>e</mi></msub> </math> is inversely associated with the trait-specific genomic relationship of a candidate to the training set. This provides an explanation for the inverse association between <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><msub><mi>Q</mi> <mi>e</mi></msub> </math> and the accuracy of prediction.",
        Data(articles=[Article(title="Genomic Prediction Accuracy", abstract=None, domain="Genomics", topic="Genomic Prediction", subtopic="QTL and LD")]),
    ),
    (
        "Magnetic information is usually stored in ferromagnets, where the ↑ and ↓ spin states are distinguishable due to time-reversal symmetry breaking. These states induce opposite signs of the Hall effect proportional to magnetization, which is widely used for their electrical read-out. By contrast, conventional antiferromagnets with a collinear antiparallel spin configuration cannot host such functions, because of <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>T</mi> <mi>t</mi></mrow> </math> symmetry (time-reversal <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mi>T</mi></math> followed by translation t symmetry) and lack of macroscopic magnetization. Here we report the experimental observation of a spontaneous Hall effect in the collinear antiferromagnet FeS at room temperature. In this compound, the ↑↓ and ↓↑ spin states induce opposite signs of the spontaneous Hall effect. Our analysis suggests that this does not reflect magnetization, but rather originates from a fictitious magnetic field associated with the <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>T</mi> <mi>t</mi></mrow> </math> -symmetry-broken antiferromagnetic order. The present results pave the way for electrical reading and writing of the ↑↓ and ↓↑ spin states in conductive systems at room temperature, and suggest that <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>T</mi> <mi>t</mi></mrow> </math> -symmetry-broken collinear antiferromagnets can serve as an information medium with vanishingly small magnetization.",
        Data(articles=[Article(title="Spontaneous Hall Effect in Antiferromagnets", abstract=None, domain="Physics", topic="Magnetism", subtopic="Antiferromagnets")]),
    ),
]

messages = []

for txt, tool_call in examples:
    if tool_call.articles:
        ai_response = "Detected articles."
    else:
        ai_response = "Detected no articles."
    messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

search_query = "math"
user_email = "your_email@example.com"
articles = fetch_pubmed_title_abstract(search_query, user_email, max_results=5)

message_no_extraction = {
    "role": "user",
    "content": print(articles),
}

structured_llm = llm.with_structured_output(schema=Data)
response = structured_llm.invoke([message_no_extraction])

print(response)