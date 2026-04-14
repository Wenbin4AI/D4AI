from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from collections import defaultdict

model_name = "Qwen/Qwen3-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map={"": 0}
)
tokenizer = AutoTokenizer.from_pretrained(model_name)



def LLM_response(prompt=""):
    if prompt == "":
        prompt = "The query triplet is (radiotherapy, hypernym, ?). The quantized representation of entity radiotherapy is: [2006] [588] [350] [1486] [214] [929] [328] [1424] [1792] [919] [944] [740] [438] [843] [147] [628] The answer candidates and corresponding quantized representations are as follows: disease, [156] [1880] [1777] [185] [121] [720] [783] [1713] [945] [1077] [180] [1576] [1574] [1433] [216] [1280] tomography, [182] [597] [657] [1486] [404] [468] [732] [564] [833] [1470] [1756] [626] [1674] [843] [1928] [513] medical care, [422] [68] [1329] [1517] [1251] [431] [1479] [1445] [1666] [407] [952] [406] [1337] [388] [1982] [685] status, [1721] [1906] [1773] [1811] [12] [892] [1625] [1476] [1561] [176] [534] [1463] [1657] [368] [70] [1618] physiological state, [1721] [718] [267] [394] [120] [1105] [885] [1823] [1496] [23] [952] [406] [1559] [1198] [1149] [1800] medical science, [565] [413] [842] [1517] [350] [873] [575] [595] [721] [935] [1554] [175] [708] [1643] [1820] [1775] infection, [565] [1594] [990] [1066] [974] [40] [434] [874] [1401] [371] [1700] [1118] [1709] [52] [71] [1408] picturing, [788] [168] [641] [1797] [927] [711] [1608] [123] [1163] [1460] [952] [406] [1752] [1464] [553] [1158] medicine, [1879] [1216] [691] [296] [1743] [892] [1851] [595] [2039] [1428] [426] [740] [399] [579] [433] [1987] unhealthiness, [1389] [644] [570] [258] [635] [647] [732] [1139] [1660] [407] [464] [1020] [1574] [1905] [926] [1971] grounds, [1268] [1053] [803] [780] [1194] [285] [328] [289] [1163] [915] [1921] [1020] [524] [1774] [430] [1572] defense reaction, [1881] [1821] [1620] [1703] [435] [995] [908] [1308] [1596] [1598] [401] [2008] [903] [817] [92] [1158] radiology, [1478] [588] [1340] [1797] [1436] [1914] [1894] [1424] [634] [1460] [1756] [740] [673] [843] [108] [1088] radioscopy, [1005] [1002] [1441] [137] [1436] [1378] [1479] [1649] [1544] [1470] [534] [626] [902] [272] [904] [1874] treat, [396] [2007] [1935] [1305] [1993] [1030] [1690] [1445] [1203] [1417] [1554] [495] [1752] [1001] [1236] [98] specialize, [1005] [1933] [1976] [780] [927] [1728] [575] [105] [1791] [1598] [616] [1118] [1752] [425] [437] [1847] therapy, [396] [816] [81] [488] [336] [1164] [1690] [1288] [900] [915] [1554] [175] [666] [1622] [765] [685] specialism, [384] [816] [599] [394] [435] [789] [1479] [105] [664] [407] [1554] [103] [1752] [1708] [697] [1130] symptom, [1721] [1913] [772] [858] [120] [1150] [1374] [289] [1666] [1417] [944] [2008] [1454] [958] [1169] [1800] medicine, [156] [350] [1599] [1955] [1368] [508] [1527] [1445] [1561] [1460] [426] [1142] [940] [653] [793] [471] Please generate quantized representations of the top-3 potential answer entities, ranked from highest to lowest:  "
    messages = [
        {"role": "system", "content": "This is a knowledge graph completion task, which needs to predict the tail entity for an incomplete query triplet."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # token_1id = tokenizer.convert_tokens_to_ids(prompt)   # tokenizer("hello")["input_ids"][1]

    # print(token_1id)
    # print(tokenizer(prompt).tokens())

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)

    return response

LLM_response()
