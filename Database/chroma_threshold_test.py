import chromadb
from pathlib import Path
from model_config import get_embedding_function, get_reranker


# ============================================================
# 1. Models
# ============================================================

embedding_function = get_embedding_function()
reranker = get_reranker()


# ============================================================
# 2. USE A SEPARATE TEST DATABASE
#    This does NOT touch your main ./chroma_db database.
# ============================================================

TEST_DB_PATH = Path(__file__).resolve().parent / "chroma_threshold_test_db"

client = chromadb.PersistentClient(path=str(TEST_DB_PATH))

# Cosine distance:
#   0   = very similar
#   larger = less similar
#
# We use cosine explicitly so the threshold is easy to reason about.
collection = client.get_or_create_collection(
    name="threshold_test",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
)


# ============================================================
# 3. TEST CORPUS
#
# We deliberately include:
#   A. Directly related
#   B. Clearly but indirectly related
#   C. Borderline / weakly related
#   D. Completely unrelated
#
# The goal is NOT simply "get the exact answer".
# The goal is to find a threshold that keeps anything that
# could reasonably help answer the question, while rejecting
# clearly outlandish material.
# ============================================================

documents = [
    # ----------------------------
    # A. DIRECTLY RELATED
    # ----------------------------

    "A compiler translates source code written in a programming language "
    "into another representation, such as machine code or bytecode.",

    "Lexical analysis is the compiler phase that converts a stream of "
    "source characters into tokens.",

    "A parser takes tokens produced by lexical analysis and determines "
    "whether they follow the grammar of the programming language.",

    "Semantic analysis checks meaning-related properties of a program, "
    "such as type compatibility, variable declarations, and scope rules.",

    "An abstract syntax tree (AST) represents the grammatical structure "
    "of source code while omitting many details of the original syntax.",

    "An intermediate representation (IR) is a representation of a program "
    "used internally by a compiler between source code and final machine code.",

    "Register allocation assigns variables or temporary values to a limited "
    "set of processor registers, often spilling some values to memory.",

    "Compiler optimization transforms a program into an equivalent form "
    "that may execute faster, use less memory, or otherwise improve performance.",

    "Dead code elimination removes computations or code paths whose results "
    "cannot affect the observable behavior of the program.",

    "Constant propagation replaces variables with known constant values "
    "when the compiler can determine those values at compile time.",


    # ----------------------------
    # B. CLEARLY / INDIRECTLY RELATED
    # ----------------------------

    "A compiler frontend commonly includes lexical analysis, parsing, and "
    "semantic analysis before the program reaches optimization and code generation.",

    "A backend takes an intermediate representation and performs tasks such as "
    "instruction selection, register allocation, scheduling, and machine-code generation.",

    "Static analysis examines source code or an intermediate representation "
    "without executing the program and can detect certain classes of errors.",

    "Type checking verifies that operations in a program are used with "
    "compatible types according to the language's type system.",

    "Control-flow graphs represent possible paths of execution through a "
    "function and are widely used in compiler optimization and analysis.",

    "Data-flow analysis computes information about how values and definitions "
    "move through the program and is used by many compiler optimizations.",

    "Machine code is the low-level instruction representation that can be "
    "executed by a particular processor architecture.",

    "JIT compilation translates code during program execution, combining "
    "ideas from interpretation and ahead-of-time compilation.",

    "A linker combines object files and libraries into an executable or "
    "another final binary representation.",

    "An assembler translates assembly language into machine-code or object-code "
    "representations.",


    # ----------------------------
    # C. BORDERLINE / WEAKLY RELATED
    # ----------------------------

    "Programming language design concerns syntax, semantics, type systems, "
    "abstraction mechanisms, and how programmers express computations.",

    "A debugger allows developers to execute programs step by step, inspect "
    "variables, and investigate unexpected behavior.",

    "Software build systems automate compilation, linking, testing, dependency "
    "management, and packaging tasks for larger projects.",

    "Operating systems manage processes, memory, files, devices, and other "
    "resources used by software applications.",

    "CPU caches reduce the average time needed to access frequently reused data "
    "by keeping it closer to the processor.",

    "Computer architecture describes processor organization, instruction sets, "
    "memory systems, and the interaction between hardware components.",

    "Version control systems track changes to source code and allow developers "
    "to collaborate safely on software projects.",

    "An integrated development environment provides editing, navigation, "
    "debugging, and other tooling for software development.",

    "Automated testing checks whether software behaves according to expected "
    "requirements and can include unit, integration, and system tests.",

    "Garbage collection automatically reclaims memory that is no longer reachable "
    "by a program.",


    # ----------------------------
    # D. CLEARLY UNRELATED
    # ----------------------------

    "Pineapples have rough brown skin and juicy yellow flesh and are often "
    "associated with tropical climates.",

    "The human heart pumps blood through the circulatory system using rhythmic "
    "contractions of cardiac muscle.",

    "The Eiffel Tower is a landmark in Paris constructed from iron and completed "
    "in the nineteenth century.",

    "Chocolate is made from processed cacao beans and is commonly used in "
    "desserts and confectionery.",

    "Rainbows are optical phenomena produced when light interacts with water "
    "droplets in the atmosphere.",

    "A soccer team generally tries to score goals by moving a ball into the "
    "opponent's goal while following the rules of the game.",

    "The Pacific Ocean is the largest ocean on Earth and covers a substantial "
    "portion of the planet's surface.",

    "Photosynthesis allows plants, algae, and some microorganisms to convert "
    "light energy into chemical energy.",

    "A mortgage is a loan commonly used to finance the purchase of real estate "
    "and is repaid over an agreed period.",

    "Baking bread typically involves mixing flour, water, yeast, and other "
    "ingredients before fermentation and baking.",
]

ids = [f"doc{i}" for i in range(1, len(documents) + 1)]


# Start clean every time so old experiments cannot affect the results.
try:
    collection.delete(ids=ids)
except Exception:
    pass

collection.add(
    ids=ids,
    documents=documents,
)


# ============================================================
# 4. TEST QUERIES
#
# Each query has expected RELATED documents.
# "Related" here means: a document could reasonably be useful
# to answering the query, even if it is not the exact answer.
#
# That is deliberately broader than exact retrieval accuracy.
# ============================================================

tests = [
    {
        "query": "What does a compiler do?",
        "expected_groups": {
            "direct": set(range(1, 11)),
            "indirect": set(range(11, 21)),
            "borderline": set(range(21, 31)),
            "unrelated": set(range(31, 41)),
        },
    },
    {
        "query": "How does lexical analysis work?",
        "expected_groups": {
            "direct": {2, 11, 12},
            "indirect": {3, 13, 14, 15, 18, 20},
            "borderline": {21, 22, 23, 27, 28, 29},
            "unrelated": set(range(31, 41)),
        },
    },
    {
        "query": "How do compilers optimize programs?",
        "expected_groups": {
            "direct": {8, 9, 10},
            "indirect": {6, 13, 15, 16, 17},
            "borderline": {21, 22, 23, 24, 25, 26, 28, 29},
            "unrelated": set(range(31, 41)),
        },
    },
    {
        "query": "What is register allocation?",
        "expected_groups": {
            "direct": {7},
            "indirect": {6, 8, 10, 16, 19, 20},
            "borderline": {24, 25, 26},
            "unrelated": set(range(31, 41)),
        },
    },
    {
        "query": "How does a compiler turn source code into machine code?",
        "expected_groups": {
            "direct": {1, 2, 3, 4, 5, 6, 7, 8},
            "indirect": {11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
            "borderline": {21, 22, 24, 25, 26, 28},
            "unrelated": set(range(31, 41)),
        },
    },
]


# ============================================================
# 5. RETRIEVE A LARGE TOP-K FIRST
#
# Important:
# We deliberately retrieve many candidates before thresholding.
# Otherwise you cannot inspect the weak tail of the ranking.
# ============================================================

INITIAL_K = 40


def get_results(query):
    results = collection.query(
        query_texts=[query],
        n_results=min(INITIAL_K, collection.count()),
        include=["documents", "distances"],
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    return list(zip(docs, distances))


# ============================================================
# 6. PRINT THE RAW DISTANCE DISTRIBUTION
#
# This is the most important part of the experiment.
# Look at where DIRECT / INDIRECT / BORDERLINE / UNRELATED
# items tend to sit.
# ============================================================

def classify_doc(doc_index):
    if doc_index <= 10:
        return "DIRECT"
    elif doc_index <= 20:
        return "INDIRECT"
    elif doc_index <= 30:
        return "BORDERLINE"
    return "UNRELATED"


def print_ranked_results(query):
    results = get_results(query)

    # Chroma returns candidates ordered by distance (lowest first).
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    for rank, (document, distance) in enumerate(results, start=1):
        # Find the corresponding document index.
        doc_index = documents.index(document) + 1
        category = classify_doc(doc_index)

        print(
            f"{rank:>2}. distance={distance:.4f} "
            f"[{category:<9}] doc{doc_index:<2} | {document}"
        )


# ============================================================
# 7. THRESHOLD EVALUATION
#
# For cosine distance:
#   keep document if distance <= threshold
#
# We measure:
#
#   RELATED RECALL:
#       Of everything that SHOULD pass, how much did we keep?
#
#   UNRELATED PASS RATE:
#       Of clearly unrelated documents, how much slipped through?
#
# Your desired threshold is likely one that gives high recall
# while keeping unrelated pass rate very low.
# ============================================================

def evaluate_threshold(query, expected_groups, threshold):
    results = get_results(query)

    expected_related = (
        expected_groups["direct"]
        | expected_groups["indirect"]
        | expected_groups["borderline"]
    )

    expected_unrelated = expected_groups["unrelated"]

    kept = []

    for document, distance in results:
        doc_index = documents.index(document) + 1

        if distance <= threshold:
            kept.append((doc_index, distance))

    kept_ids = {doc_index for doc_index, _ in kept}

    related_recall = (
        len(kept_ids & expected_related) / len(expected_related)
        if expected_related
        else 0
    )

    unrelated_pass_rate = (
        len(kept_ids & expected_unrelated) / len(expected_unrelated)
        if expected_unrelated
        else 0
    )

    return {
        "kept": kept,
        "related_recall": related_recall,
        "unrelated_pass_rate": unrelated_pass_rate,
        "kept_count": len(kept),
    }


# ============================================================
# 8. TEST A RANGE OF THRESHOLDS
#
# Adjust this range after seeing the raw distances above.
# ============================================================

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


def run_threshold_sweep():
    print("\n" + "#" * 100)
    print("THRESHOLD SWEEP")
    print("#" * 100)

    for test in tests:
        query = test["query"]
        groups = test["expected_groups"]

        print("\n" + "-" * 100)
        print(f"QUERY: {query}")
        print("-" * 100)

        print(
            f"{'Threshold':>10} | "
            f"{'Kept':>4} | "
            f"{'Related Recall':>14} | "
            f"{'Unrelated Pass':>15}"
        )
        print("-" * 65)

        for threshold in THRESHOLDS:
            result = evaluate_threshold(
                query,
                groups,
                threshold,
            )

            print(
                f"{threshold:>10.2f} | "
                f"{result['kept_count']:>4} | "
                f"{result['related_recall'] * 100:>13.1f}% | "
                f"{result['unrelated_pass_rate'] * 100:>14.1f}%"
            )


# ============================================================
# 9. SHOW WHAT A SELECTED THRESHOLD ACTUALLY KEEPS
# ============================================================

def inspect_threshold(query, threshold):
    test = next(t for t in tests if t["query"] == query)

    result = evaluate_threshold(
        query,
        test["expected_groups"],
        threshold,
    )

    print("\n" + "=" * 100)
    print(f"SELECTED THRESHOLD: {threshold:.2f}")
    print(f"QUERY: {query}")
    print("=" * 100)

    if not result["kept"]:
        print("Nothing passed the threshold.")
        return

    for doc_index, distance in result["kept"]:
        category = classify_doc(doc_index)

        print(
            f"distance={distance:.4f} "
            f"[{category:<9}] "
            f"doc{doc_index:<2} | "
            f"{documents[doc_index - 1]}"
        )


# ============================================================
# 10. OPTIONAL RERANKER EXPERIMENT
#
# This lets you inspect your proposed pipeline:
#
#   vector retrieval -> threshold -> reranker -> top K
#
# Notice that the threshold is applied BEFORE reranking here.
# ============================================================

def retrieve_with_threshold_and_rerank(query, threshold, final_k=5):
    results = collection.query(
        query_texts=[query],
        n_results=min(INITIAL_K, collection.count()),
        include=["documents", "distances"],
    )

    candidates = []

    for document, distance in zip(
        results["documents"][0],
        results["distances"][0],
    ):
        if distance <= threshold:
            candidates.append((document, distance))

    if not candidates:
        return []

    # Only rerank the documents that survived the retrieval threshold.
    pairs = [[query, document] for document, _ in candidates]
    reranker_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidates, reranker_scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return reranked[:final_k]


# ============================================================
# 11. RUN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    # First inspect the full ranking.
    for test in tests:
        print_ranked_results(test["query"])

    # Then compare thresholds across all queries.
    run_threshold_sweep()

    # Example: change this after looking at the sweep.
    EXAMPLE_THRESHOLD = 0.50

    inspect_threshold(
        "How do compilers optimize programs?",
        EXAMPLE_THRESHOLD,
    )

    # Example of the complete pipeline.
    print("\n" + "#" * 100)
    print("THRESHOLD -> RERANK -> TOP K EXAMPLE")
    print("#" * 100)

    query = "How do compilers optimize programs?"

    reranked = retrieve_with_threshold_and_rerank(
        query=query,
        threshold=EXAMPLE_THRESHOLD,
        final_k=5,
    )

    for rank, ((document, distance), score) in enumerate(
        reranked,
        start=1,
    ):
        print(
            f"{rank}. vector_distance={distance:.4f} "
            f"reranker_score={score:.4f}\n"
            f"   {document}\n"
        )
