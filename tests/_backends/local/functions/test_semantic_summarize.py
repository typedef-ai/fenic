import polars as pl

from fenic import Paragraph, col, semantic


def test_semantic_summarization(local_session):
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
        semantic.summarize(col("text"), format=Paragraph()).alias("summarized_text")
    )
    result, _ = df.collect()
    assert result.schema["summarized_text"] == pl.String
    assert len(result['summarized_text'])<120