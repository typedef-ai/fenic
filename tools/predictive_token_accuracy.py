# noqa: D100
# Safely import Anthropic packages
import anthropic
from anthropic.types import MessageParam

from fenic._inference.token_counter import TiktokenTokenCounter


def anthropic_tokenizer_compatibility(): # noqa: D103
    reference_token_counter = anthropic.Client()
    corpora = {
        "eng": """"
        Four score and seven years ago our fathers brought forth, upon this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.
    
    Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived, and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting-place for those who here gave their lives, that that nation might live. It is altogether fitting and proper that we should do this.
    
    But, in a larger sense, we can not dedicate, we can not consecrate we can not hallow this ground. The brave men, living and dead, who struggled here, have consecrated it far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here.
    
    It is for us, the living, rather, to be dedicated here to the unfinished work which they who fought here, have, thus far, so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us that from these honored dead we take increased devotion to that cause for which they here gave the last full measure of devotion that we here highly resolve that these dead shall not have died in vain that this nation, under God, shall have a new birth of freedom and that government of the people, by the people, for the people, shall not perish from the earth.
        """,

        "fr": """"
        Les représentants du peuple français, constitués en Assemblée nationale, considérant que l'ignorance, l'oubli ou le mépris des droits de l'homme sont les seules causes des malheurs publics et de la corruption des gouvernements, ont résolu d'exposer, dans une déclaration solennelle, les droits naturels, inaliénables et sacrés de l'homme, afin que cette déclaration, constamment présente à tous les membres du corps social, leur rappelle sans cesse leurs droits et leurs devoirs ; afin que les actes du pouvoir législatif, et ceux du pouvoir exécutif, pouvant être à chaque instant comparés avec le but de toute institution politique, en soient plus respectés ; afin que les réclamations des citoyens, fondées désormais sur des principes simples et incontestables, tournent toujours au maintien de la Constitution et au bonheur de tous.
        """,

        "cn-tc": """"
        中國是世界上歷史最悠久的國家之一。中國各族人民共同創造了光輝燦爛的文化，具有光榮的革命傳統。
    
    一八四○年以後，封建的中國逐漸變成半殖民地、半封建的國家。中國人民為國家獨立、民族解放和民主自由進行了前仆後繼的英勇奮鬥。
    
    二十世紀，中國發生了翻天覆地的偉大歷史變革。
    
    一九一一年孫中山先生領導的辛亥革命，廢除了封建帝制，創立了中華民國。但是，中國人民反對帝國主義和封建主義的歷史任務還沒有完成。
    
    一九四九年，以毛澤東主席為領袖的中國共產黨領導中國各族人民，在經歷了長期的艱難曲折的武裝鬥爭和其他形式的鬥爭以後，終於推翻了帝國主義、封建主義和官僚資本主義的統治，取得了新民主主義革命的偉大勝利，建立了中華人民共和國。從此，中國人民掌握了國家的權力，成為國家的主人。
    
    中華人民共和國成立以後，我國社會逐步實現了由新民主主義到社會主義的過渡。生產資料私有制的社會主義改造已經完成，人剝削人的制度已經消滅，社會主義制度已經確立。工人階級領導的、以工農聯盟為基礎的人民民主專政，實質上即無產階級專政，得到鞏固和發展。中國人民和中國人民解放軍戰勝了帝國主義、霸權主義的侵略、破壞和武裝挑釁，維護了國家的獨立和安全，增強了國防。經濟建設取得了重大的成就，獨立的、比較完整的社會主義工業體系已經基本形成，農業生產顯著提高。教育、科學、文化等事業有了很大的發展，社會主義思想教育取得了明顯的成效。廣大人民的生活有了較大的改善。
        """,
        "cn-sc": """
           中国是世界上历史最悠久的国家之一。中国各族人民共同创造了光辉灿烂的文化，具有光荣的革命传统。

一八四○年以后，封建的中国逐渐变成半殖民地、半封建的国家。中国人民为国家独立、民族解放和民主自由进行了前仆后继的英勇奋斗。

二十世纪，中国发生了翻天覆地的伟大历史变革。

一九一一年孙中山先生领导的辛亥革命，废除了封建帝制，创立了中华民国。但是，中国人民反对帝国主义和封建主义的历史任务还没有完成。

一九四九年，以毛泽东主席为领袖的中国共产党领导中国各族人民，在经历了长期的艰难曲折的武装斗争和其他形式的斗争以后，终于推翻了帝国主义、封建主义和官僚资本主义的统治，取得了新民主主义革命的伟大胜利，建立了中华人民共和国。从此，中国人民掌握了国家的权力，成为国家的主人。
        """,
        "jp": """
        日本国民は、正当に選挙された国会における代表者を通じて行動し、われらとわれらの子孫のために、諸国民との協和による成果と、わが国全土にわたつて自由のもたらす恵沢を確保し、政府の行為によつて再び戦争の惨禍が起ることのないやうにすることを決意し、ここに主権が国民に存することを宣言し、この憲法を確定する。そもそも国政は、国民の厳粛な信託によるものであつて、その権威は国民に由来し、その権力は国民の代表者がこれを行使し、その福利は国民がこれを享受する。これは人類普遍の原理であり、この憲法は、かかる原理に基くものである。われらは、これに反する一切の憲法、法令及び詔勅を排除する。
        """,
        "tool" : """"
{"messages": [{"content": "You've won a free iPhone! Click here", "role": "user"}], "model": "claude-3-5-haiku-latest", "system": [{"text": "You are a text classification expert. Classify the following document into one of the following labels: spam, not spam. Respond with *only* the predicted label.", "type": "text", "cache_control": {"type": "ephemeral"}}], "tool_choice": {"name": "output_formatter", "type": "tool"}, "tools": [{"name": "output_formatter", "input_schema": {"$defs": {"EmailType": {"enum": ["spam", "not spam"], "title": "EmailType", "type": "string"}}, "properties": {"output": {"$ref": "#/$defs/EmailType"}}, "required": ["output"], "title": "EnumModel", "type": "object"}, "description": "Format the output of the model to correspond strictly to the provided schema.", "cache_control": {"type": "ephemeral"}}]}        """
    }
    haiku_model_name = "claude-3-5-haiku-latest"
    o200k_token_counter= TiktokenTokenCounter(haiku_model_name, fallback_encoding="o200k_base")
    cl100k_token_counter = TiktokenTokenCounter(haiku_model_name, fallback_encoding="cl100k_base")
    o200k_fudge_factors = {}
    cl100k_fudge_factors = {}
    for label, corpus in corpora.items():
        trimmed_corpus = corpus.strip()
        messages = [
            {"role": "user", "content": trimmed_corpus},
            {"role": "assistant", "content": trimmed_corpus},
        ]
        o200k_tokenizer_count = float(o200k_token_counter.count_tokens(messages))
        cl100k_tokenizer_count = float(cl100k_token_counter.count_tokens(messages))
        haiku_token_count = reference_token_counter.messages.count_tokens(messages=convert_messages_anthropic(messages),
                                                                          model=haiku_model_name)
        o200k_fudge_factor = haiku_token_count.input_tokens / o200k_tokenizer_count
        cl100k_fudge_factor = haiku_token_count.input_tokens / cl100k_tokenizer_count

        print(f"corpus: {label} -- ff: o200k: {o200k_fudge_factor}, cl100k: {cl100k_fudge_factor}")
        o200k_fudge_factors[label] = o200k_fudge_factor
        cl100k_fudge_factors[label] = cl100k_fudge_factor
    print(
        f"o200k: {sum(o200k_fudge_factors.values()) / len(o200k_fudge_factors)} -- {o200k_fudge_factors}, cl100k: {sum(cl100k_fudge_factors.values()) / len(cl100k_fudge_factors)} -- {cl100k_fudge_factors}")

def convert_messages_anthropic(messages: list[dict[str, str]]) -> list[MessageParam]: # noqa: D103
    return [MessageParam(content=message["content"], role=message["role"]) for message in messages]