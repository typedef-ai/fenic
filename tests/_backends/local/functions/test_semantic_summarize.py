
import polars as pl
import pytest

from fenic import KeyPoints, col, semantic
from fenic.api.session import OpenAIModelConfig, SemanticConfig, Session, SessionConfig
from fenic.core.error import ValidationError


def test_semantic_summarization_default_case(local_session):
    source = local_session.create_dataframe(
        {
            "text": [

                '''
                Zanthor quibbed the drindle when parloons flustered in the glimber dusk. Trindles of bloop-swirled marquent vines danced over the skizzle groves as the farnip hooted in the glim-glam sky. Nobody knew why the dursh wobbled when the glorp sang to the seventh pindle of Nareth.
                Jibbleflap thirteen exponded across the sprangle moons, while the whorplet took a gander at the snozzle. Twirp! went the dingleberry boats, floating on the jarnwax of forgotten smoogleton dreams. Every plarnish tharned its blibble like it was the quazzening of Zarthok Prime.
                Meanwhile, deep beneath the stobbled mound, krintworms skadaddled in rhythmic plumps. Wibbling and snorfling, they carried the flarnik dust to the upper greebs where zorgles whispered truths only known to the Slarn Council of Seven Wumps. Shadoomp! came the trumpet of the holy Frabblehonk, sending ripples through the puddlecrete basin.
                "Do not squancht the glorb!" shouted old man Trunckle, shaking his mopple-sheen in the air. But it was too late—Thrennik had already eaten the last gibberhoop.
                Oodeling snoffs dropped from the sky like spangled confetti, covering the yawnplugs in a shiny glaze of mumbo spritz. Drongle-kins leapt from zip-zip pods, waving their flabbles and screaming "Boojaboo!" at passing stinkle cats.
                Grav-flutes resonated in harmonic chuzz, setting the stage for the cosmic glibberfest. No one remembered the prophecy of the eighth zindle, nor did they care. Floop. Wankle. Jibjab.
                The moon howled like a crumpled crebbox, and the stars hiccupped in binary clusters of ploff. Somewhere, in the heart of the unfathomed zarksea, the binglethrob finally exhaled.
                Thus ended the great blarnicle of Zingo-Zango and the thrice-unspoken glibberwomp.
                ''',
            ],
        }
    )

    df = source.select(
        semantic.summarize(col("text")).alias("summarized_text")
    )
    result = df.to_polars()
    assert result.schema["summarized_text"] == pl.String
    assert len(result['summarized_text'])<120
    assert len(result['summarized_text'])>0


def test_semantic_summarization_keypoints(local_session):
    source = local_session.create_dataframe(
        {
            "text": [

                '''
                It is a truth universally acknowledged, that a single man in possession of a good fortune,
                must be in want of a wife. However little known the feelings or views of such a man may be
                on his first entering a neighbourhood, this truth is so well fixed in the minds of the
                surrounding families, that he is considered the rightful property of some one or other of
                their daughters. “My dear Mr. Bennet,” said his lady to him one day, “have you heard that
                Netherfield Park is let at last?” Mr. Bennet replied that he had not. “But it is,” returned
                she; “for Mrs. Long has just been here, and she told me all about it.” Mr. Bennet made no
                answer. “Do not you want to know who has taken it?” cried his wife impatiently. “You want
                to tell me, and I have no objection to hearing it.” This was invitation enough. “Why, my
                dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large
                fortune from the north of England; that he came down on Monday in a chaise and four to
                see the place, and was so much delighted with it, that he agreed with Mr. Morris
                immediately; he is to take possession before Michaelmas, and some of his servants are to
                be in the house by the end of next week.” “What is his name?” “Bingley.” “Is he married
                or single?” “Oh! single, my dear, to be sure! A single man of large fortune; four or five
                thousand a year. What a fine thing for our girls!” “How so? how can it affect them?”
                “My dear Mr. Bennet,” replied his wife, “how can you be so tiresome! You must know that
                I am thinking of his marrying one of them.” “Is that his design in settling here?” “Design!
                nonsense, how can you talk so! But it is very likely that he may fall in love with one of
                them, and therefore you must visit him as soon as he comes.” “I see no occasion for that.
                You and the girls may go, or you may send them by themselves, which perhaps will
                still be better, for as you are as handsome as any of them, Mr. Bingley might like you
                the best of the party.” “My dear, you flatter me. I certainly have had my share of beauty,
                but I do not pretend to be anything extraordinary now. When a woman has five grown-up daughters,
                she ought to give over thinking of her own beauty.” “In such cases, a woman has not often much
                beauty to think of.” “But, my dear, you must indeed go and see Mr. Bingley when he comes into
                the neighbourhood.” “It is more than I engage for, I assure you.” “But consider your daughters.
                Only think what an establishment it would be for one of them. Sir William and Lady Lucas are
                determined to go, merely on that account; for in general, you know, they visit no new-comers.
                Indeed, you must go, for it will be impossible for us to visit him, if you do not.”
                ''',
            ],
        }
    )
    df = source.select(
        semantic.summarize(col("text"), format=KeyPoints(max_points = 10)).alias("summarized_text")
    )
    result = df.to_polars()
    assert result.schema["summarized_text"] == pl.String
    result_lines: list[str] = result['summarized_text'][0].splitlines()
    # should not exceed 10 key points
    assert sum(1 for line in result_lines if line.strip().startswith("- ") or line.strip().startswith("* ")) <= 10
    assert sum(1 for line in result['summarized_text'][0].splitlines() if line.strip().startswith("- ") or line.strip().startswith("* ")) >=1

def test_semantic_summarize_without_models():
    """Test that an error is raised if no language models are configured."""
    session_config = SessionConfig(
        app_name="semantic_summarize_without_models",
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"text": ["hello"]}).select(semantic.summarize(col("text")).alias("summarized_text"))
    session.stop()

    session_config = SessionConfig(
        app_name="semantic_summarize_without_models",
        semantic=SemanticConfig(
            embedding_models={"oai-small": OpenAIModelConfig(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)},
        ),
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"text": ["hello"]}).select(semantic.summarize(col("text")).alias("summarized_text"))
    session.stop()
