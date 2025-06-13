import os
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import faiss
import difflib

# Load Gemini API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Expanded book catalog
books = [
    # Romance
    {"title": "Pride and Prejudice", "genre": "romance", "year": "19th century", "description": "Classic romantic novel by Jane Austen.", "price": 12},
    {"title": "Me Before You", "genre": "romance", "year": "21st century", "description": "Modern love story by Jojo Moyes.", "price": 13},
    {"title": "The Notebook", "genre": "romance", "year": "20th century", "description": "Romantic drama by Nicholas Sparks.", "price": 14},
    {"title": "Outlander", "genre": "romance", "year": "20th century", "description": "Historical romance mixed with time travel.", "price": 15},

    # Fantasy
    {"title": "Harry Potter", "genre": "fantasy", "year": "21st century", "description": "Fantasy novel about a young wizard.", "price": 20},
    {"title": "The Hobbit", "genre": "fantasy", "year": "20th century", "description": "Fantasy novel by J.R.R. Tolkien.", "price": 18},
    {"title": "The Lord of the Rings", "genre": "fantasy", "year": "20th century", "description": "Epic high fantasy trilogy by Tolkien.", "price": 22},
    {"title": "Mistborn", "genre": "fantasy", "year": "21st century", "description": "Epic fantasy series by Brandon Sanderson.", "price": 19},

    # Horror
    {"title": "Dracula", "genre": "horror", "year": "19th century", "description": "Classic vampire novel by Bram Stoker.", "price": 12},
    {"title": "Frankenstein", "genre": "horror", "year": "19th century", "description": "Classic horror novel by Mary Shelley.", "price": 11},
    {"title": "It", "genre": "horror", "year": "20th century", "description": "Horror novel by Stephen King.", "price": 16},
    {"title": "The Shining", "genre": "horror", "year": "20th century", "description": "Psychological horror novel by Stephen King.", "price": 16},

    # Drama
    {"title": "To Kill a Mockingbird", "genre": "drama", "year": "20th century", "description": "Novel about racial injustice.", "price": 13},
    {"title": "The Catcher in the Rye", "genre": "drama", "year": "20th century", "description": "Coming-of-age novel by J.D. Salinger.", "price": 14},
    {"title": "A Thousand Splendid Suns", "genre": "drama", "year": "21st century", "description": "Story set in Afghanistan by Khaled Hosseini.", "price": 15},
    {"title": "The Kite Runner", "genre": "drama", "year": "21st century", "description": "Story about friendship and redemption.", "price": 15},

    # Dystopian
    {"title": "1984", "genre": "dystopian", "year": "20th century", "description": "Totalitarian dystopian novel by George Orwell.", "price": 15},
    {"title": "Brave New World", "genre": "dystopian", "year": "20th century", "description": "Science fiction dystopia by Aldous Huxley.", "price": 14},
    {"title": "Fahrenheit 451", "genre": "dystopian", "year": "20th century", "description": "Novel about censorship by Ray Bradbury.", "price": 13},
    {"title": "The Handmaid's Tale", "genre": "dystopian", "year": "20th century", "description": "Dystopian oppression novel by Margaret Atwood.", "price": 14},

    # Others
    {"title": "The Iliad", "genre": "epic", "year": "8th century BC", "description": "Greek epic poem by Homer.", "price": 10},
    {"title": "The Art of War", "genre": "strategy", "year": "5th century BC", "description": "Military treatise by Sun Tzu.", "price": 11},
    {"title": "The Great Gatsby", "genre": "classic", "year": "20th century", "description": "American classic by F. Scott Fitzgerald.", "price": 14}
]

# Prepare embedding text for each book
for book in books:
    book["embedding_text"] = f"Title: {book['title']}. Genre: {book['genre']}. Year: {book['year']}. Description: {book['description']}."

# Gemini embedding function
def get_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return np.array(response['embedding'], dtype="float32")

# Generate embeddings for all books
for book in books:
    book["embedding_vector"] = get_embedding(book["embedding_text"])

# MAIN APP START
print("\n📚 Welcome to the AI Bookstore!\n")

# Display available genres before search
available_genres = sorted(set(b['genre'].lower() for b in books))
print("📚 Available genres:")
for g in available_genres:
    print(f"- {g.capitalize()}")
print()

while True:
    genre_input = input("Please enter a genre you're interested in: ").strip().lower()

    # Filter first by genre (strict filtering)
    filtered_books = [b for b in books if b['genre'].lower() == genre_input]

    if not filtered_books:
        print(f"\n⚠ No books found for genre '{genre_input}'. Please try again.\n")
        continue

    # Build FAISS index on filtered dataset
    filtered_embeddings = np.array([b['embedding_vector'] for b in filtered_books])
    dimension = filtered_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(filtered_embeddings)

    # Embed user query for vector search
    query = f"Books about {genre_input}"
    query_embedding = get_embedding(query).reshape(1, -1)
    top_k = min(5, len(filtered_books))
    distances, indices = index.search(query_embedding, top_k)

    # Display recommendations
    print("\n🔎 AI Recommendations:\n")
    recommended_books = []
    for i, idx in enumerate(indices[0]):
        book = filtered_books[idx]
        recommended_books.append(book)
        print(f"{i+1}. {book['title']} ({book['year']})\n   {book['description']}\n   Price: ${book['price']}\n")

    # Book selection logic
    title_lookup = {b["title"].lower(): b for b in recommended_books}

    while True:
        selection = input("\nPlease type which book you'd like to buy (title or number): ").strip().lower()

        if selection.isdigit():
            selection_num = int(selection)
            if 1 <= selection_num <= len(recommended_books):
                chosen_book = recommended_books[selection_num - 1]
                break

        all_titles = list(title_lookup.keys())
        if selection in all_titles:
            chosen_book = title_lookup[selection]
            break

        matches = difflib.get_close_matches(selection, all_titles, n=1, cutoff=0.5)
        if matches:
            chosen_book = title_lookup[matches[0]]
            break

        print("AI: Please enter valid book title or number from recommendations.")

    # Confirmation
    print(f"\n📖 You selected: {chosen_book['title']}")
    print(f"Price: ${chosen_book['price']}")
    confirm = input("Do you want to confirm your purchase? (yes/no): ").lower()

    if confirm == "yes":
        print("\n✅ Purchase confirmed. Thank you for your order!\n")
    else:
        print("\n❌ Purchase cancelled.\n")

    break
