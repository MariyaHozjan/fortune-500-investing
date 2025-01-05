
import streamlit as st
import pandas as pd
from models.promethee import promethee
from main import preference_functions, criteria, weights

from main import data, results, results_WSM, results_AHP_eigenvector, results_AHP_geometric, results_AHP_normalized, results_topsis, results_promethee_linear

#st.title("Fortune 500 investiranje")

# Add a logo and force it to the top of the sidebar
st.sidebar.markdown(
    """
    <style>
        [data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            justify-content: flex-start;
            gap: 10px;
        }
        [data-testid="stSidebar"] > div:first-child img {
            height: 50px;
        }
    </style>
    <div>
        <img src="https://via.placeholder.com/50" alt="Logo">
        <h2 style="margin: 0; color: white;">Fortune 500</h2>
    </div>
    """,
    unsafe_allow_html=True
)


st.header("Fortune 500 investiranje", divider=True)
#st.markdown("<hr style='border-top: 2px solid #EC0ED8;'>", unsafe_allow_html=True)
filtered_data = None

def page_1():
    st.title("Naši podatki")

    # Initialize session state for selected companies if not set
    if 'selected_companies' not in st.session_state:
        st.session_state['selected_companies'] = data['Name'].unique().tolist()

    # Multiselect box for selecting companies
    selected_companies = st.multiselect(
        "Izberi podjetja za analizo:",
        options=data['Name'].unique(),
        default=st.session_state['selected_companies']
    )

    # Update session state with the current selection
    st.session_state['selected_companies'] = selected_companies

    # Filter data based on the selected companies
    filtered_data = data[data['Name'].isin(selected_companies)]

    st.dataframe(filtered_data, use_container_width=True, hide_index=True)
    return filtered_data


def page_2():
    st.title("WSM")

    st.write("Spodaj so prikazani rezultati metode WSM, za vsako od podjetij med katerimi se odločamo. Večja kot je številka, boljši je rezultat.")
    st.info(' Z klikom na stolpca lahko spreminjaš razvrstitev.', icon="🖋️")

    st.dataframe(results_WSM, use_container_width=True)

    st.divider()

    st.write("Spodaj so rezultati prikazani še vizualno za boljšo predstavo.")
    st.bar_chart(results_WSM, use_container_width=True, color="#6FD6FF")

    st.divider()
    url = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4464669"
    st.write("Več o WSM metodi si lahko pogledaš [tukaj](%s)" % url)

def page_3():
    st.title("AHP")

    st.write("Spodaj so prikazani rezultati metode AHP, za vsako od podjetij med katerimi se odločamo. Večja kot je številka, boljši je rezultat.")
    st.info(' Z klikom na stolpca lahko spreminjaš razvrstitev.', icon="🖋️")

    method = st.selectbox(
        "Izberi metodo za izračun AHP:",
        ("Eigenvector", "Normalized Column Sum", "Geometric Mean")
    )

    # Display the corresponding results based on the selected method
    if method == "Eigenvector":
        st.dataframe(results_AHP_eigenvector, use_container_width=True)
    elif method == "Normalized Column Sum":
        st.dataframe(results_AHP_normalized, use_container_width=True)
    else:  # Geometric Mean
        st.dataframe(results_AHP_geometric, use_container_width=True)

    st.divider()

    st.write("Spodaj so rezultati prikazani še vizualno za boljšo predstavo.")

    # Add a bar chart for the selected method's results
    if method == "Eigenvector":
        st.bar_chart(results_AHP_eigenvector, use_container_width=True)
    elif method == "Normalized Column Sum":
        st.bar_chart(results_AHP_normalized, use_container_width=True)
    else:  # Geometric Mean
        st.bar_chart(results_AHP_geometric, use_container_width=True)

    st.divider()
    url = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4464669"
    st.write("Več o AHP metodi si lahko pogledaš [tukaj](%s)" % url)

def page_4():
    st.title("Topsis")

    st.write("Spodaj so prikazani rezultati metode Topsis, za vsako od podjetij med katerimi se odločamo. Večja kot je številka, boljši je rezultat.")
    st.info('Z klikom na stolpca lahko spreminjaš razvrstitev.', icon="🖋️")

    st.dataframe(results_topsis, use_container_width=True)

    st.divider()

    st.write("Spodaj so rezultati prikazani še vizualno za boljšo predstavo.")
    st.bar_chart(results_topsis, use_container_width=True, color="#6FD6FF")

    st.divider()
    url = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4464669"
    st.write("Več o Topsis metodi si lahko pogledaš [tukaj](%s)" % url)

def page_5():
    st.title("PROMETHEE")

    st.write("Spodaj so prikazani rezultati metode PROMETHEE, za vsako od podjetij med katerimi se odločamo.")

    # Select preference function type
    preference_fn_type = st.selectbox(
        "Izberi preferenčno funkcijo:",
        ("Linear", "Usual", "U-Shape", "V-Shape", "Level", "Gaussian")
    )

    # Select whether to maximize or minimize
    maximize = st.radio("Optimizacija:", ("Maksimiziraj", "Minimiziraj"))
    maximize = True if maximize == "Maksimiziraj" else False

    # Create two columns
    col1, col2 = st.columns(2)

    # Slider for parameter p in the first column
    with col1:
        p = st.slider(
            "Select the value of p:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,  # Default value
            step=0.1
        )

    # Slider for parameter q in the second column
    with col2:
        q = st.slider(
            "Select the value of q:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,  # Default value
            step=0.1
        )

    st.write(f"Selected values: p = {p}, q = {q}")

    # Generate the selected preference function array
    p_fn = preference_functions[preference_fn_type](maximize)
    # Call the PROMETHEE method with the selected preference function
    scores_promethee_dynamic = promethee(criteria.values, p_fn, weights)

    # Create a dataframe for the results
    results_promethee_dynamic = pd.DataFrame({
        'PROMETHEE': scores_promethee_dynamic
    }, index=data['Name'])

    # Display the dataframe
    st.dataframe(results_promethee_dynamic, use_container_width=True)

    st.divider()

    st.write("Spodaj so rezultati prikazani še vizualno za boljšo predstavo.")
    st.bar_chart(results_promethee_dynamic, use_container_width=True)

    st.divider()

    url = "https://en.wikipedia.org/wiki/PROMETHEE"
    st.write("Več o PROMETHEE metodi si lahko pogledaš [tukaj](%s)" % url)

def page_6():
    st.title("Rezultati metod")
    st.write("Spodaj so za vsako od uporabljenih metod prikazana 3 podjetja z najboljšimi rezultati.")
    top_companies_wsm = pd.concat([
        results["WSM"].nlargest(3).rename("Score")
    ], axis=1)

    top_companies_topsis = pd.concat([
        results["TOPSIS"].nlargest(3).rename("Score")
    ], axis=1)

    top_companies_ahp = pd.concat([
        results["AHP"].nlargest(3).rename("Score")
    ], axis=1)

    top_companies_promethee = pd.concat([
        results["PROMETHEE"].nlargest(3).rename("Score")
    ], axis=1)

    st.subheader("WSM")
    st.dataframe(top_companies_wsm, width=350)
    st.subheader("AHP")
    st.dataframe(top_companies_ahp, width=350)
    st.subheader("Topsis")
    st.dataframe(top_companies_topsis, width=350)
    st.subheader("Promethee")
    st.dataframe(top_companies_promethee, width=350)

    st.divider()

    st.subheader("Kam se nam toraj najbolj splača investirati?")
    st.write("Glede na dobljene rezultate bi nam bilo najbolj smiselno investirati v **Berkshire Hathaway**, sej je podjetje v vseh metodah zavzelo prvo mesto. Če pa si želimo svojo investicijo diverzificirati pa lahko del denarja vložimo tudi v **Apple** in **Amazon**.")

#pg = st.navigation([st.Page(page_1), st.Page(page_2), st.Page(page_3), st.Page(page_4), st.Page(page_5), st.Page(page_6)])
pages = {
    "Podatki": [
        st.Page(page_1, title="Tabela podatkov")
    ],
    "Modeli": [
        st.Page(page_2, title="WSM"),
        st.Page(page_3, title="AHP"),
        st.Page(page_4, title="Topsis"),
        st.Page(page_5, title="Promethee")
    ],
    "Kam investirati?": [
        st.Page(page_6, title="Rezultati")
    ]
}

pg = st.navigation(pages)
pg.run()