import chromadb
from pathlib import Path
from model_config import get_embedding_function, get_reranker


embedding_function = get_embedding_function()
reranker = get_reranker()

TEST_DB_PATH = Path(__file__).resolve().parent / "embedding_test_db"
client = chromadb.PersistentClient(path=str(TEST_DB_PATH))

collection = client.get_or_create_collection(
    name="pdf_documents_test",
    embedding_function=embedding_function,
)

collection.add(
    ids=["id1", "id2", "id3", "id4", "id5",
        "id6", "id7", "id8", "id9", "id10",
        "id11", "id12", "id13", "id14", "id15",
        "id16", "id17", "id18", "id19", "id20"],

    documents=[
        # 1
        "Pineapple has a rough brown shell on the outside and juicy yellow flesh inside. It is sweet, tropical, and sometimes controversial as a pizza topping.",

        # 2
        "Oranges are citrus fruits with bright orange skin and orange-colored flesh. They are juicy, slightly acidic, and commonly eaten fresh or turned into juice.",

        # 3
        "Bananas are usually yellow when ripe, green before ripening, and develop dark spots as they age. Their flesh is pale and soft, with a naturally sweet taste.",

        # 4
        "Tomatoes are botanically fruits even though they are often treated like vegetables in cooking. They have red skin, red flesh, and contain small edible seeds.",

        # 5
        "Apples can have red, green, or yellow skin depending on the variety. They are crisp, firm, mildly sweet, and usually much less juicy than citrus fruits.",

        # 6
        "Lemons are yellow citrus fruits with intensely sour juice. Their yellow peel surrounds pale yellow flesh and they are often used to add acidity to food and drinks.",

        # 7
        "Limes are small green citrus fruits with tart juice. They are commonly associated with cocktails, Mexican food, and dishes where a sharp acidic flavor is wanted.",

        # 8
        "Strawberries are small red berries with tiny seeds visible on their surface. The inside is generally pale red or pink, and they have a sweet and slightly acidic flavor.",

        # 9
        "Watermelons have a thick green rind and juicy red flesh. They contain many dark seeds in traditional varieties and are strongly associated with hot summer weather.",

        # 10
        "Blueberries are small dark blue or purple berries. Their flesh is much lighter than their skin and they contain antioxidants and small edible seeds.",

        # 11
        "Avocados have rough green or nearly black skin and soft green flesh surrounding one large pit. Unlike most sweet fruits, they have a creamy texture and relatively mild flavor.",

        # 12
        "Coconuts have a hard brown shell and white flesh inside. The fruit also contains coconut water and is commonly used in tropical cooking and desserts.",

        # 13
        "Peaches have fuzzy orange-yellow or reddish skin and soft, juicy yellow-orange flesh. A single large stone sits in the center.",

        # 14
        "Grapes grow in clusters and can be green, red, or dark purple. They are small, juicy fruits and may be eaten fresh or dried into raisins.",

        # 15
        "Kiwifruit has a fuzzy brown exterior but vivid green flesh inside, usually with many tiny black seeds. Its flavor is sweet, tangy, and slightly acidic.",

        # 16
        "Pineapple is one of the fruits people frequently debate putting on pizza. Some enjoy the sweet and salty combination with ham, while others strongly dislike it.",

        # 17
        "A traditional savory pizza might contain tomato sauce, cheese, mushrooms, peppers, onions, and meat. Sweet fruit toppings are considered unusual by some people.",

        # 18
        "A vegetarian pizza avoids meat but can contain vegetables, mushrooms, cheese, tomato sauce, and various herbs. It is suitable for someone who does not eat meat.",

        # 19
        "Citrus fruits such as oranges, lemons, and limes are characterized by juicy segments and acidic flavors. They are commonly associated with vitamin C.",

        # 20
        "Among common fruits, the banana changes appearance dramatically during ripening: green when unripe, yellow when ripe, and increasingly brown or black as it becomes overripe.",
    ],

    metadatas=[
        *[{"source": "fruit-reference.pdf", "page": 1} for _ in range(20)],
    ],
)

queries = [
    # Direct semantic matches
    "Which fruit has a brown exterior and yellow flesh?",
    "What fruit is orange both outside and inside?",
    "Which fruit becomes yellow when it is ripe?",
    "Which fruit is technically classified as a fruit but commonly used as a vegetable?",
    "Which fruit is extremely sour and yellow?",
    
    # Indirect / paraphrased
    "What fruit has a tropical appearance with rough skin and golden flesh?",
    "I'm looking for something acidic that is commonly squeezed into drinks.",
    "Which fruit has a creamy interior and one huge seed?",
    "What fruit has a fuzzy skin and a stone in the middle?",
    "Which fruit looks completely different on the outside compared with its interior?",
    
    # Semantic similarity
    "Tell me about something people argue about putting on pizza.",
    "Which fruit would cause the biggest pizza topping debate?",
    "What food combines sweetness with a salty pizza topping?",
    "Which pizza option contains no meat?",
    "What kind of pizza would someone who avoids meat choose?",
    
    # Citrus reasoning
    "Name a fruit from the acidic vitamin-C family.",
    "Which fruits are related to oranges in the citrus category?",
    "I'm looking for a small green fruit with a sour taste.",
    "Which yellow fruit is much more sour than an orange?",
    
    # Color-based retrieval
    "Which fruit has dark skin but much lighter flesh?",
    "What fruit has green skin and green flesh around a large pit?",
    "Which fruit has a green rind but red flesh?",
    "What fruit is dark blue on the outside but lighter inside?",
    
    # Tricky / overlapping
    "Which fruit changes from green to yellow and eventually dark brown?",
    "Which fruit has visible tiny black seeds in bright green flesh?",
    "Which fruit has seeds on its skin rather than primarily inside?",
    "Which fruit is commonly dried to make raisins?",
    
    # Conceptual
    "Which fruit is creamy rather than particularly sweet?",
    "Which fruit is associated with summer because it contains extremely juicy red flesh?",
    "Which fruits would be considered unusual toppings for a traditional savory pizza?",
]

expected = {
    "Which fruit has a brown exterior and yellow flesh?":
        ["id1"],

    "What fruit is orange both outside and inside?":
        ["id2"],

    "Which fruit becomes yellow when it is ripe?":
        ["id3", "id20"],

    "Which fruit is technically classified as a fruit but commonly used as a vegetable?":
        ["id4"],

    "Which fruit is extremely sour and yellow?":
        ["id6"],

    "What fruit has a creamy interior and one huge seed?":
        ["id11"],

    "What fruit has a fuzzy skin and a stone in the middle?":
        ["id13"],

    "Tell me about something people argue about putting on pizza.":
        ["id16"],

    "What kind of pizza would someone who avoids meat choose?":
        ["id18"],

    "Name a fruit from the acidic vitamin-C family.":
        ["id19"],

    "Which fruit has green skin and green flesh around a large pit?":
        ["id11"],

    "Which fruit has a green rind but red flesh?":
        ["id9"],

    "What fruit is dark blue on the outside but lighter inside?":
        ["id10"],

    "Which fruit changes from green to yellow and eventually dark brown?":
        ["id20"],

    "Which fruit has visible tiny black seeds in bright green flesh?":
        ["id15"],

    "Which fruit is commonly dried to make raisins?":
        ["id14"],
}

correct_count = 0

hits = 0
k = 3

for query, expected_ids in expected.items():

    results = collection.query(
        query_texts=[query],
        n_results=10
    )

    documents = results["documents"][0]
    ids = results["ids"][0]

    pairs = [
        [query, document]
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(ids, scores),
        key=lambda x: x[1],
        reverse=True
    )

    retrieved = [id_ for id_, score in ranked[:k]]

    hit = any(
        doc_id in retrieved
        for doc_id in expected_ids
    )

    print(
        f"{'✅' if hit else '❌'} "
        f"{query}\n"
        f"   Expected: {expected_ids}\n"
        f"   Retrieved: {retrieved}\n"
    )

    if hit:
        hits += 1



accuracy = hits / len(expected) * 100

print(f"\nRecall@{k}: {accuracy:.1f}%")

accuracy = (correct_count / len(queries)) * 100

print("\n" + "=" * 50)
print(f"Accuracy: {correct_count}/{len(queries)} ({accuracy:.2f}%)")
