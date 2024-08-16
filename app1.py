import streamlit as st
import streamlit.components.v1 as components
import os
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from neo4j import GraphDatabase
from pyvis.network import Network

# Set page config
st.set_page_config(page_title="LLM Graph Processing App", layout="wide", page_icon=":chart_with_upwards_trend:")

# Introduction about the app
st.title("LLM Graph Processing App")
st.markdown("""
    Welcome to the LLM Graph Processing App! This application allows you to generate and visualize knowledge graphs from Wikipedia data. 
    By leveraging the power of Large Language Models (LLMs) and Neo4j, you can explore the relationships between entities 
    in a visual and interactive way.
    
    To get started, please provide the necessary configuration details in the sidebar, enter the Wikipedia page you want to analyze, 
    and click 'Process and Generate Graph'. The app will process the Wikipedia content, create a graph, and display it for your exploration.
""")

# Sidebar for user inputs
st.sidebar.header("Configuration")
st.sidebar.write("Please provide your credentials and Wikipedia query below:")

# User inputs
wikipedia_query = st.sidebar.text_input("Wikipedia Page Name", "Elizabeth I")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
neo4j_uri = st.sidebar.text_input("Neo4j URI")
neo4j_username = st.sidebar.text_input("Neo4j Username", "neo4j")
neo4j_password = st.sidebar.text_input("Neo4j Password", type="password")

# Default parameters
chunk_size = 1000
chunk_overlap = 100
max_documents = 3

def create_graph_visualization(cypher_result):
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.force_atlas_2based()
    
    for record in cypher_result:
        source = record['s']
        target = record['t']
        relationship = record['r']
        
        # Add nodes without labels
        net.add_node(str(source.id), title=str(dict(source)), color="#4CAF50")
        net.add_node(str(target.id), title=str(dict(target)), color="#2196F3")
        
        # Add edge with relationship type
        net.add_edge(str(source.id), str(target.id), label=str(relationship.type), title=str(relationship.type))
    
    # Set network options for better visualization
    net.set_options("""
    var options = {
      "nodes": {
        "shape": "dot",
        "size": 20
      },
      "edges": {
        "font": {"size": 10, "align": "middle"},
        "color": "lightgray",
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      }
    }
    """)
    
    return net.generate_html()

if st.button("Process and Generate Graph"):
    try:
        # Set up the environment
        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["NEO4J_URI"] = neo4j_uri
        os.environ["NEO4J_USERNAME"] = neo4j_username
        os.environ["NEO4J_PASSWORD"] = neo4j_password

        # Initialize LLM and graph transformer
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
        llm_transformer = LLMGraphTransformer(llm=llm)

        # Load and process documents
        with st.spinner("Loading Wikipedia documents..."):
            loader = WikipediaLoader(query=wikipedia_query)
            raw_documents = loader.load()
            
            # Filter out summary sections
            filtered_documents = [
                doc for doc in raw_documents 
                if 'summary' not in doc.metadata.get('section_title', '').lower()
            ]
            st.success(f"Loaded {len(filtered_documents)} documents from Wikipedia")

        with st.spinner("Splitting documents..."):
            text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            documents = text_splitter.split_documents(filtered_documents[:max_documents])
            st.success(f"Split into {len(documents)} chunks")

        with st.spinner("Converting to graph documents..."):
            graph_documents = llm_transformer.convert_to_graph_documents(documents)
            st.success(f"Converted to {len(graph_documents)} graph documents")

        # Initialize Neo4j graph and add documents
        with st.spinner("Adding documents to Neo4j graph..."):
            graph = Neo4jGraph()
            graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
            st.success("Added documents to Neo4j graph")

        # Query the graph and visualize
        with st.spinner("Querying graph and generating visualization..."):
            cypher_query = "MATCH (s)-[r]->(t) RETURN s,r,t LIMIT 50"
            driver = GraphDatabase.driver(
                uri=neo4j_uri,
                auth=(neo4j_username, neo4j_password)
            )
            
            with driver.session() as session:
                # Check Neo4j content
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()['node_count']
                st.write(f"Number of nodes in Neo4j: {node_count}")

                # Run the main query
                result = session.run(cypher_query)
                result_list = list(result)
                if not result_list:
                    st.warning("The query returned no results. The graph may be empty.")
                else:
                    html_content = create_graph_visualization(result_list)
                    components.html(html_content, height=600)
                    st.success("Graph visualization created successfully")

        st.success("Graph processing and visualization complete!")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.info("Enter your configuration in the sidebar and click 'Process and Generate Graph' to start.")
