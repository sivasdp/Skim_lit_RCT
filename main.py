import streamlit as st
import preprocess_text
import tensorflow as tf
import loaded_model
import pandas as pd

st.sidebar.header("Department of Artificial Intelligence and Machine Learning")
st.sidebar.title("III Year")

st.sidebar.title("Team members:")
st.sidebar.write('''

Sivadhandapani S (720721115050) (TL).

Balakumar K (720721115011)

Ramachandiran M (720721115041)
                 
Gunalan R (720721115016)
                 

''')


st.sidebar.title("Tech Stack Used:")
st.sidebar.write('''

1)Tensorflow (for Modeling)
                 
2)Streamlit (for deployment)

3)Python (Programming Language)

4)Mlflow (for tracking,monitoring,versioning the models)
                 

''')
st.sidebar.title("Contact Information:")
st.sidebar.write('''

Phone:+91 7603880048.
Email: sdpsiva191@gmail.com       

''')
# Title and description
st.title("SkimLit 📄⚡")
st.header("An Advanced level NLP Model to classifiy Abstract of the Medical Research Papers!")


# Text area for user input
paragraph = st.text_area("Enter your Abstract here:", height=200)

# paragraph = "The recently released GPT-4 Code Interpreter has demonstrated remarkable proficiency in solving challenging math problems, primarily attributed to its ability to seamlessly reason with natural language, generate code, execute code, and continue reasoning based on the execution output. In this paper, we present a method to fine-tune open-source language models, enabling them to use code for modeling and deriving math equations and, consequently, enhancing their mathematical reasoning abilities. We propose a method of generating novel and high-quality datasets with math problems and their code-based solutions, referred to as MathCodeInstruct. Each solution interleaves natural language, code, and execution results. We also introduce a customized supervised fine-tuning and inference approach. This approach yields the MathCoder models, a family of models capable of generating code-based solutions for solving challenging math problems. Impressively, the MathCoder models achieve state-of-the-art scores among open-source LLMs on the MATH (45.2%) and GSM8K (83.9%) datasets, substantially outperforming other open-source alternatives. Notably, the MathCoder model not only surpasses ChatGPT-3.5 and PaLM-2 on GSM8K and MATH but also outperforms GPT-4 on the competition-level MATH dataset."

# Create a Streamlit button
process_button = st.button("Process Abstract....")
if process_button:
    df, test_sentences = preprocess_text.split_paragraph_to_dataframe(paragraph)

    test_chars = [preprocess_text.split_chars(sentence) for sentence in test_sentences]

    test_line_numbers_one_hot, test_total_lines_one_hot = preprocess_text.one_hot(df,tf)

    x = (test_line_numbers_one_hot,
         test_total_lines_one_hot,
         tf.constant(test_sentences),
         tf.constant(test_chars))

    predicted_class = loaded_model.loaded_model(x,tf)
    df['Predicted class'] = predicted_class
    df.drop(['Line number','Total lines'], axis=1,inplace=True)
    # st.write(df)

    lines=df['Sentence Content']
    pred=df["Predicted class"]
    objective = ''
    background = ''
    method = ''
    conclusion = ''
    result = ''
    for i, line in enumerate(lines):
        if pred[i] == 'OBJECTIVE':
            objective = objective + line
        
        elif pred[i] == 'BACKGROUND':
            background = background + line
        
        elif pred[i] == 'METHODS':
            method = method + line
        
        elif pred[i] == 'RESULTS':
            result = result + line
        
        elif pred[i] == 'CONCLUSIONS':
            conclusion = conclusion + line
        else:
            print("error")
    
    st.markdown(f'### Objective:')
    st.write(f'{objective}')
    st.markdown(f'### Background:')
    st.write(f'{background}')
    st.markdown(f'### Methods:')
    st.write(f'{method}')
    st.markdown(f'### Result:')
    st.write(f'{result}')
    st.markdown(f'### Conclusion:')
    st.write(f'{conclusion}')

example_button=st.button("Input Examples")
if example_button:
    example_data=pd.read_csv("example.csv")
    abstract_data=example_data['abstract']
    for i in range(len(abstract_data)) :
        st.write(f"{i}.{abstract_data[i]}\n")

