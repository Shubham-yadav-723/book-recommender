# Core Pkg
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image

img = Image.open("app_logo.png")
st.set_page_config(page_title="LibGeniee", page_icon=img)



# Load EDA

# for fetching the posters


def fetch_poster(book_url):
    url = "{}".format(book_url)
    return url


# Load books dataset
books_dict = pickle.load(open('books_2.pkl', 'rb'))
books = pd.DataFrame(books_dict)

# importing similarity file
# cosine_sim_corpus = pickle.load(open('similarity.pkl', 'rb'))


@st.cache()
def corpus_recommendations(title):
    books['corpus'] = (pd.Series(books[['authors', 'tag_name']]
                                 .fillna('')
                                 .values.tolist()
                                 ).str.join(' '))

    tf_corpus = TfidfVectorizer(analyzer='word', ngram_range=(
        1, 2), min_df=0, stop_words='english')
    tfidf_matrix_corpus = tf_corpus.fit_transform(books['corpus'])
    cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)

    titles = books[['title', 'image_url']]
    indices = pd.Series(books.index, index=books['title'])

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:10]
    book_indices = [i[0] for i in sim_scores]
    recommended_books = []
    recommended_posters = []
    for i in sim_scores:
        book_url = titles.iloc[i[0]].image_url
        recommended_posters.append(fetch_poster(book_url))
        recommended_books.append(titles.iloc[i[0]].title)

    return recommended_books, recommended_posters


# Load Our Dataset
def load_data(data):
    df = pd.read_csv(data)
    return df


# Fxn
# Vectorize + Cosine Similarity Matrix

def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    # Get the cosine
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat


# Recommendation Sys
@st.cache
def get_recommendation(title, cosine_sim_mat, df, num_of_rec=10):
    # indices of the course
    course_indices = pd.Series(
        df.index, index=df['course_title']).drop_duplicates()
    # Index of course
    idx = course_indices[title]

    # Look into the cosine matr for that index
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:]]
    selected_course_scores = [i[0] for i in sim_scores[1:]]

    # Get the dataframe & title
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = selected_course_scores
    final_recommended_courses = result_df[[
        'course_title', 'similarity_score', 'url', 'price', 'num_subscribers']]
    return final_recommended_courses.head(num_of_rec)


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>

</div>
"""

# Search For Course


@st.cache
def search_term_if_not_found(term, df):
    result_df = df[df['course_title'].str.contains(term)]
    return result_df


def main():

    st.title("LibGeniee")
    st.text("A study material Recommendation system")

    menu = ["Home", "Recommend Course", "Recommend Books", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    df = load_data("udemy_courses.csv")

    if choice == "Home":
        st.subheader("Home")
        st.image("app_logo.png", width=370)

    elif choice == "Recommend Course":
        st.subheader("Recommend Courses")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
        search_term = st.selectbox(
            "Select the Course", df['course_title'].values)
        num_of_rec = st.sidebar.number_input(
            "Number of Recommendations you want ?", 4, 30, 7)
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    results = get_recommendation(
                        search_term, cosine_sim_mat, df, num_of_rec)
                    with st.expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)

                    for row in results.iterrows():
                        rec_title = row[1][0]
                        rec_score = row[1][1]
                        rec_url = row[1][2]
                        rec_price = row[1][3]
                        rec_num_sub = row[1][4]

                        # st.write("Title",rec_title,)
                        stc.html(RESULT_TEMP.format(rec_title, rec_score,
                                 rec_url, rec_price, rec_num_sub), height=350)
                except:
                    results = "Not Found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found(search_term, df)
                    st.dataframe(result_df)

                # How To Maximize Your Profits Options Trading
    elif choice == "Recommend Books":
        st.subheader("Recommend Books")
        search_book = st.selectbox(
            "Select the book", books['original_title'].values)

        if st.button("Recommend"):
            # recommendation = corpus_recommendations(search_book)
            # st.subheader("Recommended Books")
            # for i in recommendation:

            #     st.write(i)
            recommended_books, recommended_posters = corpus_recommendations(
                search_book)
            col1, col2, col3, = st.columns(
                3)
            col4, col5, col6 = st.columns(
                3)
            col7, col8, col9 = st.columns(
                3)
            with col1:
                st.text(recommended_books[0])
                st.image(recommended_posters[0], width=200)
            with col2:
                st.text(recommended_books[1])
                st.image(recommended_posters[1], width=200)

            with col3:
                st.text(recommended_books[2])
                st.image(recommended_posters[2], width=200)
            with col4:
                st.text(recommended_books[3])
                st.image(recommended_posters[3], width=200)
            with col5:
                st.text(recommended_books[4])
                st.image(recommended_posters[4], width=200)
            with col6:
                st.text(recommended_books[5])
                st.image(recommended_posters[5], width=200)
            with col7:
                st.text(recommended_books[6])
                st.image(recommended_posters[6], width=200)
            with col8:
                st.text(recommended_books[7])
                st.image(recommended_posters[7], width=200)
            with col9:
                st.text(recommended_books[8])
                st.image(recommended_posters[8], width=200)

    else:
        st.subheader("Details")
        st.text(" shubham yadav")

        st.text("Email: Shubham724262@gmail.com")
        st.image("app_logo.png", width=250)


if __name__ == '__main__':
    main()
