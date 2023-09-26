import spacy

def setup_spacy():
    # Check if the desired language model is already installed
    if not spacy.util.is_package("en_core_web_sm"):
        # Download and install the language model (without specifying a version)
        spacy.cli.download("en_core_web_sm")
        spacy.cli.link("en_core_web_sm", "en", force=True)

# Call the setup function when the script is run
if __name__ == "__main__":
    setup_spacy()
