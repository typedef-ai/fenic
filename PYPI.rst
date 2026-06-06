fenic: The dataframe (re)built for LLM inference
################################################

fenic is an opinionated, PySpark-inspired DataFrame framework for building AI and agentic applications. Transform unstructured and structured data into insights using familiar DataFrame operations enhanced with semantic intelligence. With first-class support for markdown, transcripts, and semantic operators, plus efficient batch inference across any model provider.

See `github.com/typedef-ai/fenic <https://www.github.com/typedef-ai/fenic>`_ for more information

Install optional feature extras only when you need heavier operators:

.. code-block:: bash

   pip install "fenic[pdf]"       # semantic.parse_pdf and PDF metadata loading
   pip install "fenic[cluster]"   # DataFrame.semantic.with_cluster_labels
   pip install "fenic[sim-join]"  # semantic.sim_join

Extras can be combined with model provider extras, for example:

.. code-block:: bash

   pip install "fenic[google,pdf,cluster,sim-join]"
