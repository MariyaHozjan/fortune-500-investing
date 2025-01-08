from typing import List, Optional
import numpy as np
import streamlit as st
import pandas as pd
from dataclasses import dataclass
from models.topsis import topsis
from models.wsm import wsm
from models.ahp import ahp
from models.promethee import promethee, make_linear_fn, make_usual_fn, make_u_shape_fn, make_v_shape_fn, make_level_fn, \
    make_gaussian_fn
from utils.normalization import max_norm

st.set_page_config(layout="wide")

@dataclass()
class PrometheeParameters:
    function_name: str
    maximize: bool
    q: Optional[float]
    p: Optional[float]
    s: Optional[float]

    def make_preference_function(self):
        if self.function_name == "Linear":
            return make_linear_fn(self.maximize, q=self.q, p=self.p)
        elif self.function_name == "Usual":
            return make_usual_fn(self.maximize)
        elif self.function_name == "U-Shape":
            return make_u_shape_fn(self.maximize, q=self.q)
        elif self.function_name == "V-Shape":
            return make_v_shape_fn(self.maximize, p=self.p)
        elif self.function_name == "Level":
            return make_level_fn(self.maximize, q=self.q, p=self.p)
        elif self.function_name == "Gaussian":
            return make_gaussian_fn(self.maximize, s=self.s)

        raise RuntimeError("Unknown preference function.")

@dataclass
class AHPParameters:
    method: str

@dataclass
class Context:
    data: pd.DataFrame
    selected_columns: List[str]
    selected_companies: List[str]
    weights: pd.DataFrame
    ahp_params: AHPParameters
    promethee_params: PrometheeParameters # maps column to optimization parameters


    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.selected_columns = ['Revenues ($M)', 'Profits ($M)', 'Assets ($M)']
        self.selected_companies = data.index
        self.weights = pd.Series(np.ones(len(self.data.columns)) / len(self.selected_columns), index=self.data.columns, name="Weight")
        self.ahp_params = AHPParameters(method="Eigenvector")
        self.promethee_params = PrometheeParameters(function_name="Linear",maximize=True,q=0.1,p=0.3,s=0.5)


    def filtered_data(self):
        return data.loc[self.selected_companies, self.selected_columns]

    def filtered_weights(self):
        return self.weights[self.weights.index.isin(self.selected_columns)].to_numpy()

    def compute_wsm(self):
        return pd.DataFrame(wsm(max_norm(self.filtered_data()), self.filtered_weights()), columns=["Score"])

    def compute_ahp(self):
        preprocessed_data = max_norm(self.filtered_data())
        if self.ahp_params.method == "Eigenvector":
            scores = ahp(preprocessed_data, self.filtered_weights(), method=1)
        elif self.ahp_params.method == "Normalized Column Sum":
            scores = ahp(preprocessed_data, self.filtered_weights(), method=2)
        else:
            scores = ahp(preprocessed_data, self.filtered_weights(), method=3)

        return pd.DataFrame({
            'Score': scores.values,  # Align indices
        }, index=preprocessed_data.index)

    def compute_topsis(self):
        preprocessed_data = max_norm(self.filtered_data())
        scores = topsis(preprocessed_data, self.filtered_weights())
        return pd.DataFrame({
            'Score': scores.values,  # Align indices
        }, index=preprocessed_data.index)

    def compute_promethee(self):
        preference_fns = [self.promethee_params.make_preference_function() for _ in self.selected_columns]

        # Call the PROMETHEE method with the selected preference function
        preprocessed_data = max_norm(self.filtered_data())
        scores = promethee(max_norm(self.filtered_data()).to_numpy(), preference_fns, self.filtered_weights())

        # Create a dataframe for the results
        return pd.DataFrame({
            'Score': scores,  # Align indices
        }, index=preprocessed_data.index)

# Load dataset
file_path = './data/data.csv'
data = pd.read_csv(file_path, index_col="Name")

# Clean dataset: Check for and handle missing or invalid values
if data.isnull().any().any():
    print("Warning: The dataset contains missing values. Filling them with 0.")
    data = data.fillna(data.mean())

if 'context' not in st.session_state:
    st.session_state['context'] = Context(data=data)


st.header("Fortune 500 investiranje", divider=True)
filtered_data = None

def page_1():
    st.title("Naši podatki")

    context = st.session_state.context

    # Multiselect box for selecting attributes
    context.selected_columns = st.multiselect(
        "Izberi atribute za analizo:",
        options=context.data.columns,
        default=context.selected_columns
    )

    # Multiselect box for selecting companies
    context.selected_companies = st.multiselect(
        "Izberi podjetja za analizo:",
        options=context.data.index,
        default=context.selected_companies,
    )

    st.dataframe(context.filtered_data(), use_container_width=True)

    st.divider()
    st.header("Uteži")

    cols = st.columns(4)
    for j, column in enumerate(context.selected_columns):
        with cols[j]:
            context.weights[column] = st.number_input(key=f"{column}",
                              label=f"{column}",
                              min_value=0.0,
                              max_value=1.0,
                              value=context.weights[column])

    weights_sum = round(context.weights[context.selected_columns].sum(), 2)
    if weights_sum != 1.0:
        st.warning(f"Uteži niso veljavnje. Seštevek uteži ({weights_sum}) ni enak 1.0.")
    else:
        st.success(f"Uteži so veljavne.")

def page_2():
    st.title("WSM")

    st.write("Spodaj so prikazani rezultati metode WSM, za vsako od podjetij med katerimi se odločamo. Večja kot je številka, boljši je rezultat.")
    st.info(' Z klikom na stolpca lahko spreminjaš razvrstitev.', icon="🖋️")

    context = st.session_state.context
    results = context.compute_wsm()

    st.dataframe(results, use_container_width=True)

    st.divider()

    st.write("Spodaj so rezultati prikazani še vizualno za boljšo predstavo.")
    st.bar_chart(results, use_container_width=True, color="#6FD6FF")

def page_3():
    st.title("AHP")

    st.write("Spodaj so prikazani rezultati metode AHP, za vsako od podjetij med katerimi se odločamo. Večja kot je številka, boljši je rezultat.")
    st.info(' Z klikom na stolpca lahko spreminjaš razvrstitev.', icon="🖋️")

    context = st.session_state.context
    context.ahp_params.method = st.selectbox(
        "Izberi metodo za izračun AHP:",
        ("Eigenvector", "Normalized Column Sum", "Geometric Mean")
    )

    context = st.session_state.context
    results = context.compute_ahp()

    # Display the corresponding results based on the selected method
    st.dataframe(results, use_container_width=True)

    st.divider()

    st.write("Spodaj so rezultati prikazani še vizualno za boljšo predstavo.")

    # Add a bar chart for the selected method's results
    st.bar_chart(results, use_container_width=True)

def page_4():
    st.title("Topsis")

    st.write("Spodaj so prikazani rezultati metode Topsis, za vsako od podjetij med katerimi se odločamo. Večja kot je številka, boljši je rezultat.")
    st.info('Z klikom na stolpca lahko spreminjaš razvrstitev.', icon="🖋️")

    context = st.session_state.context
    results = context.compute_topsis()

    st.dataframe(results, use_container_width=True)

    st.divider()

    st.write("Spodaj so rezultati prikazani še vizualno za boljšo predstavo.")
    st.bar_chart(results, use_container_width=True, color="#6FD6FF")

def page_5():
    st.title("PROMETHEE")
    st.write("Spodaj so prikazani rezultati metode PROMETHEE, za vsako od podjetij med katerimi se odločamo.")

    context = st.session_state.context

    # Select preference function type
    context.promethee_params.function_name = st.selectbox(
        "Izberi preferenčno funkcijo:",
        ("Linear", "Usual", "U-Shape", "V-Shape", "Level", "Gaussian")
    )

    # Select whether to maximize or minimize
    context.promethee_params.maximize = st.radio("Optimizacija:", (True, False), index=0 if context.promethee_params.maximize else 1, format_func=lambda v: "Maksimiziraj" if v is True else "Minimiziraj")

    # Create two columns
    col1, col2 = st.columns(2)

    # Slider for parameter p in the first column
    with (col1):
        if context.promethee_params.function_name in {"Linear", "U-Shape", "Level"}:
            context.promethee_params.q = st.number_input(key=f"promethee_q",
                                label=f"Določi vrednost q:",
                                min_value=0.0,
                                max_value=1.0,
                                value=context.promethee_params.q)

        if context.promethee_params.function_name == "Gaussian":
            context.promethee_params.s = st.number_input(key=f"promethee_s",
                                label=f"Določi vrednost s:",
                                min_value=0.0,
                                max_value=1.0,
                                value=context.promethee_params.s)

        if context.promethee_params.function_name == "V-Shape":
            context.promethee_params.p = st.number_input(key=f"promethee_p",
                                    label=f"Določi vrednost p:",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=context.promethee_params.p)

    # Slider for parameter q in the second column
    with col2:
        if context.promethee_params.function_name in {"Linear", "Level"}:
            context.promethee_params.p = st.number_input(key=f"promethee_p",
                                label=f"Določi vrednost p:",
                                min_value=0.0,
                                max_value=1.0,
                                value=context.promethee_params.p)

    if (context.promethee_params.function_name in {"Linear", "Level"} and
            context.promethee_params.p <= context.promethee_params.q):
        st.error("Vrednost **p** mora bit večja od **q**.")
        return

    st.divider()

    results = context.compute_promethee()

    # Display the dataframe
    st.dataframe(results, use_container_width=True)

    st.divider()

    st.write("Spodaj so rezultati prikazani še vizualno za boljšo predstavo.")
    st.bar_chart(results, use_container_width=True)

def page_6():
    st.title("Rezultati metod")
    st.write("Spodaj so za vsako od uporabljenih metod prikazana 3 podjetja z najboljšimi rezultati.")

    context = st.session_state.context

    top_companies_wsm = pd.concat([
        context.compute_wsm().nlargest(3, columns="Score")
    ], axis=1)

    top_companies_topsis = pd.concat([
        context.compute_topsis().nlargest(3, columns="Score")
    ], axis=1)

    top_companies_ahp = pd.concat([
        context.compute_ahp().nlargest(3, columns="Score")
    ], axis=1)

    top_companies_promethee = pd.concat([
        context.compute_promethee().nlargest(3, columns="Score")
    ], axis=1)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("WSM")
        st.dataframe(top_companies_wsm, width=350)
    with col2:
        st.subheader("AHP")
        st.dataframe(top_companies_ahp, width=350)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Topsis")
        st.dataframe(top_companies_topsis, width=350)
    with col2:
        st.subheader("Promethee")
        st.dataframe(top_companies_promethee, width=350)

    st.divider()

    st.subheader("Kam se nam toraj najbolj splača investirati?")
    st.write("Glede na dobljene rezultate bi nam bilo najbolj smiselno investirati v **JP Morgan Chase**, sej je podjetje v vseh metodah zavzelo prvo mesto. Če pa si želimo svojo investicijo diverzificirati pa lahko del denarja vložimo tudi v **Berkshire Hathaway** in **Bank of America**.")

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