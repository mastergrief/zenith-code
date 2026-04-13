"""
CALM Country knowledge backend — capitals, codes, currencies, regions.

Models hallucinate capitals, mix up ISO codes, confuse currencies.
Data as of 2025. Covers 195 UN member states + common territories.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

# (capital, iso2, iso3, currency_code, currency_name, calling_code, region, population_approx)
_COUNTRIES = {
    "afghanistan": ("Kabul", "AF", "AFG", "AFN", "Afghani", "+93", "Asia", 41_129_000),
    "albania": ("Tirana", "AL", "ALB", "ALL", "Lek", "+355", "Europe", 2_837_000),
    "algeria": ("Algiers", "DZ", "DZA", "DZD", "Dinar", "+213", "Africa", 45_606_000),
    "andorra": ("Andorra la Vella", "AD", "AND", "EUR", "Euro", "+376", "Europe", 80_000),
    "angola": ("Luanda", "AO", "AGO", "AOA", "Kwanza", "+244", "Africa", 36_685_000),
    "antigua and barbuda": ("St. John's", "AG", "ATG", "XCD", "East Caribbean Dollar", "+1-268", "Americas", 94_000),
    "argentina": ("Buenos Aires", "AR", "ARG", "ARS", "Peso", "+54", "Americas", 46_655_000),
    "armenia": ("Yerevan", "AM", "ARM", "AMD", "Dram", "+374", "Asia", 2_780_000),
    "australia": ("Canberra", "AU", "AUS", "AUD", "Australian Dollar", "+61", "Oceania", 26_439_000),
    "austria": ("Vienna", "AT", "AUT", "EUR", "Euro", "+43", "Europe", 9_105_000),
    "azerbaijan": ("Baku", "AZ", "AZE", "AZN", "Manat", "+994", "Asia", 10_413_000),
    "bahamas": ("Nassau", "BS", "BHS", "BSD", "Bahamian Dollar", "+1-242", "Americas", 410_000),
    "bahrain": ("Manama", "BH", "BHR", "BHD", "Dinar", "+973", "Asia", 1_485_000),
    "bangladesh": ("Dhaka", "BD", "BGD", "BDT", "Taka", "+880", "Asia", 172_954_000),
    "barbados": ("Bridgetown", "BB", "BRB", "BBD", "Barbados Dollar", "+1-246", "Americas", 282_000),
    "belarus": ("Minsk", "BY", "BLR", "BYN", "Belarusian Ruble", "+375", "Europe", 9_200_000),
    "belgium": ("Brussels", "BE", "BEL", "EUR", "Euro", "+32", "Europe", 11_686_000),
    "belize": ("Belmopan", "BZ", "BLZ", "BZD", "Belize Dollar", "+501", "Americas", 410_000),
    "benin": ("Porto-Novo", "BJ", "BEN", "XOF", "CFA Franc", "+229", "Africa", 13_353_000),
    "bhutan": ("Thimphu", "BT", "BTN", "BTN", "Ngultrum", "+975", "Asia", 782_000),
    "bolivia": ("Sucre", "BO", "BOL", "BOB", "Boliviano", "+591", "Americas", 12_224_000),
    "bosnia and herzegovina": ("Sarajevo", "BA", "BIH", "BAM", "Convertible Mark", "+387", "Europe", 3_211_000),
    "botswana": ("Gaborone", "BW", "BWA", "BWP", "Pula", "+267", "Africa", 2_630_000),
    "brazil": ("Brasilia", "BR", "BRA", "BRL", "Real", "+55", "Americas", 216_422_000),
    "brunei": ("Bandar Seri Begawan", "BN", "BRN", "BND", "Brunei Dollar", "+673", "Asia", 449_000),
    "bulgaria": ("Sofia", "BG", "BGR", "BGN", "Lev", "+359", "Europe", 6_520_000),
    "burkina faso": ("Ouagadougou", "BF", "BFA", "XOF", "CFA Franc", "+226", "Africa", 22_674_000),
    "burundi": ("Gitega", "BI", "BDI", "BIF", "Burundian Franc", "+257", "Africa", 13_238_000),
    "cabo verde": ("Praia", "CV", "CPV", "CVE", "Escudo", "+238", "Africa", 599_000),
    "cambodia": ("Phnom Penh", "KH", "KHM", "KHR", "Riel", "+855", "Asia", 16_946_000),
    "cameroon": ("Yaounde", "CM", "CMR", "XAF", "CFA Franc", "+237", "Africa", 28_647_000),
    "canada": ("Ottawa", "CA", "CAN", "CAD", "Canadian Dollar", "+1", "Americas", 40_098_000),
    "central african republic": ("Bangui", "CF", "CAF", "XAF", "CFA Franc", "+236", "Africa", 5_742_000),
    "chad": ("N'Djamena", "TD", "TCD", "XAF", "CFA Franc", "+235", "Africa", 18_279_000),
    "chile": ("Santiago", "CL", "CHL", "CLP", "Peso", "+56", "Americas", 19_493_000),
    "china": ("Beijing", "CN", "CHN", "CNY", "Yuan Renminbi", "+86", "Asia", 1_425_671_000),
    "colombia": ("Bogota", "CO", "COL", "COP", "Peso", "+57", "Americas", 52_085_000),
    "comoros": ("Moroni", "KM", "COM", "KMF", "Comorian Franc", "+269", "Africa", 837_000),
    "congo": ("Brazzaville", "CG", "COG", "XAF", "CFA Franc", "+242", "Africa", 6_107_000),
    "congo dr": ("Kinshasa", "CD", "COD", "CDF", "Congolese Franc", "+243", "Africa", 102_262_000),
    "costa rica": ("San Jose", "CR", "CRI", "CRC", "Colon", "+506", "Americas", 5_213_000),
    "croatia": ("Zagreb", "HR", "HRV", "EUR", "Euro", "+385", "Europe", 3_855_000),
    "cuba": ("Havana", "CU", "CUB", "CUP", "Cuban Peso", "+53", "Americas", 11_194_000),
    "cyprus": ("Nicosia", "CY", "CYP", "EUR", "Euro", "+357", "Europe", 1_260_000),
    "czech republic": ("Prague", "CZ", "CZE", "CZK", "Koruna", "+420", "Europe", 10_828_000),
    "czechia": ("Prague", "CZ", "CZE", "CZK", "Koruna", "+420", "Europe", 10_828_000),
    "denmark": ("Copenhagen", "DK", "DNK", "DKK", "Krone", "+45", "Europe", 5_910_000),
    "djibouti": ("Djibouti", "DJ", "DJI", "DJF", "Djiboutian Franc", "+253", "Africa", 1_121_000),
    "dominica": ("Roseau", "DM", "DMA", "XCD", "East Caribbean Dollar", "+1-767", "Americas", 73_000),
    "dominican republic": ("Santo Domingo", "DO", "DOM", "DOP", "Peso", "+1-809", "Americas", 11_229_000),
    "ecuador": ("Quito", "EC", "ECU", "USD", "US Dollar", "+593", "Americas", 18_190_000),
    "egypt": ("Cairo", "EG", "EGY", "EGP", "Egyptian Pound", "+20", "Africa", 112_717_000),
    "el salvador": ("San Salvador", "SV", "SLV", "USD", "US Dollar", "+503", "Americas", 6_364_000),
    "equatorial guinea": ("Malabo", "GQ", "GNQ", "XAF", "CFA Franc", "+240", "Africa", 1_715_000),
    "eritrea": ("Asmara", "ER", "ERI", "ERN", "Nakfa", "+291", "Africa", 3_748_000),
    "estonia": ("Tallinn", "EE", "EST", "EUR", "Euro", "+372", "Europe", 1_373_000),
    "eswatini": ("Mbabane", "SZ", "SWZ", "SZL", "Lilangeni", "+268", "Africa", 1_202_000),
    "ethiopia": ("Addis Ababa", "ET", "ETH", "ETB", "Birr", "+251", "Africa", 126_527_000),
    "fiji": ("Suva", "FJ", "FJI", "FJD", "Fiji Dollar", "+679", "Oceania", 936_000),
    "finland": ("Helsinki", "FI", "FIN", "EUR", "Euro", "+358", "Europe", 5_563_000),
    "france": ("Paris", "FR", "FRA", "EUR", "Euro", "+33", "Europe", 64_756_000),
    "gabon": ("Libreville", "GA", "GAB", "XAF", "CFA Franc", "+241", "Africa", 2_389_000),
    "gambia": ("Banjul", "GM", "GMB", "GMD", "Dalasi", "+220", "Africa", 2_706_000),
    "georgia": ("Tbilisi", "GE", "GEO", "GEL", "Lari", "+995", "Asia", 3_729_000),
    "germany": ("Berlin", "DE", "DEU", "EUR", "Euro", "+49", "Europe", 84_482_000),
    "ghana": ("Accra", "GH", "GHA", "GHS", "Cedi", "+233", "Africa", 33_476_000),
    "greece": ("Athens", "GR", "GRC", "EUR", "Euro", "+30", "Europe", 10_341_000),
    "grenada": ("St. George's", "GD", "GRD", "XCD", "East Caribbean Dollar", "+1-473", "Americas", 126_000),
    "guatemala": ("Guatemala City", "GT", "GTM", "GTQ", "Quetzal", "+502", "Americas", 17_843_000),
    "guinea": ("Conakry", "GN", "GIN", "GNF", "Guinean Franc", "+224", "Africa", 14_191_000),
    "guinea-bissau": ("Bissau", "GW", "GNB", "XOF", "CFA Franc", "+245", "Africa", 2_106_000),
    "guyana": ("Georgetown", "GY", "GUY", "GYD", "Guyanese Dollar", "+592", "Americas", 813_000),
    "haiti": ("Port-au-Prince", "HT", "HTI", "HTG", "Gourde", "+509", "Americas", 11_725_000),
    "honduras": ("Tegucigalpa", "HN", "HND", "HNL", "Lempira", "+504", "Americas", 10_433_000),
    "hungary": ("Budapest", "HU", "HUN", "HUF", "Forint", "+36", "Europe", 9_597_000),
    "iceland": ("Reykjavik", "IS", "ISL", "ISK", "Krona", "+354", "Europe", 383_000),
    "india": ("New Delhi", "IN", "IND", "INR", "Indian Rupee", "+91", "Asia", 1_428_628_000),
    "indonesia": ("Jakarta", "ID", "IDN", "IDR", "Rupiah", "+62", "Asia", 277_534_000),
    "iran": ("Tehran", "IR", "IRN", "IRR", "Rial", "+98", "Asia", 89_172_000),
    "iraq": ("Baghdad", "IQ", "IRQ", "IQD", "Iraqi Dinar", "+964", "Asia", 44_496_000),
    "ireland": ("Dublin", "IE", "IRL", "EUR", "Euro", "+353", "Europe", 5_194_000),
    "israel": ("Jerusalem", "IL", "ISR", "ILS", "Shekel", "+972", "Asia", 9_364_000),
    "italy": ("Rome", "IT", "ITA", "EUR", "Euro", "+39", "Europe", 58_761_000),
    "ivory coast": ("Yamoussoukro", "CI", "CIV", "XOF", "CFA Franc", "+225", "Africa", 28_874_000),
    "jamaica": ("Kingston", "JM", "JAM", "JMD", "Jamaican Dollar", "+1-876", "Americas", 2_826_000),
    "japan": ("Tokyo", "JP", "JPN", "JPY", "Yen", "+81", "Asia", 123_295_000),
    "jordan": ("Amman", "JO", "JOR", "JOD", "Jordanian Dinar", "+962", "Asia", 11_337_000),
    "kazakhstan": ("Astana", "KZ", "KAZ", "KZT", "Tenge", "+7", "Asia", 19_621_000),
    "kenya": ("Nairobi", "KE", "KEN", "KES", "Kenyan Shilling", "+254", "Africa", 55_100_000),
    "kiribati": ("Tarawa", "KI", "KIR", "AUD", "Australian Dollar", "+686", "Oceania", 131_000),
    "kosovo": ("Pristina", "XK", "XKX", "EUR", "Euro", "+383", "Europe", 1_873_000),
    "kuwait": ("Kuwait City", "KW", "KWT", "KWD", "Kuwaiti Dinar", "+965", "Asia", 4_310_000),
    "kyrgyzstan": ("Bishkek", "KG", "KGZ", "KGS", "Som", "+996", "Asia", 6_974_000),
    "laos": ("Vientiane", "LA", "LAO", "LAK", "Kip", "+856", "Asia", 7_529_000),
    "latvia": ("Riga", "LV", "LVA", "EUR", "Euro", "+371", "Europe", 1_831_000),
    "lebanon": ("Beirut", "LB", "LBN", "LBP", "Lebanese Pound", "+961", "Asia", 5_490_000),
    "lesotho": ("Maseru", "LS", "LSO", "LSL", "Loti", "+266", "Africa", 2_306_000),
    "liberia": ("Monrovia", "LR", "LBR", "LRD", "Liberian Dollar", "+231", "Africa", 5_419_000),
    "libya": ("Tripoli", "LY", "LBY", "LYD", "Libyan Dinar", "+218", "Africa", 6_888_000),
    "liechtenstein": ("Vaduz", "LI", "LIE", "CHF", "Swiss Franc", "+423", "Europe", 39_000),
    "lithuania": ("Vilnius", "LT", "LTU", "EUR", "Euro", "+370", "Europe", 2_832_000),
    "luxembourg": ("Luxembourg City", "LU", "LUX", "EUR", "Euro", "+352", "Europe", 660_000),
    "madagascar": ("Antananarivo", "MG", "MDG", "MGA", "Ariary", "+261", "Africa", 30_326_000),
    "malawi": ("Lilongwe", "MW", "MWI", "MWK", "Kwacha", "+265", "Africa", 20_931_000),
    "malaysia": ("Kuala Lumpur", "MY", "MYS", "MYR", "Ringgit", "+60", "Asia", 34_308_000),
    "maldives": ("Male", "MV", "MDV", "MVR", "Rufiyaa", "+960", "Asia", 521_000),
    "mali": ("Bamako", "ML", "MLI", "XOF", "CFA Franc", "+223", "Africa", 22_594_000),
    "malta": ("Valletta", "MT", "MLT", "EUR", "Euro", "+356", "Europe", 535_000),
    "marshall islands": ("Majuro", "MH", "MHL", "USD", "US Dollar", "+692", "Oceania", 42_000),
    "mauritania": ("Nouakchott", "MR", "MRT", "MRU", "Ouguiya", "+222", "Africa", 4_862_000),
    "mauritius": ("Port Louis", "MU", "MUS", "MUR", "Mauritian Rupee", "+230", "Africa", 1_300_000),
    "mexico": ("Mexico City", "MX", "MEX", "MXN", "Mexican Peso", "+52", "Americas", 128_901_000),
    "micronesia": ("Palikir", "FM", "FSM", "USD", "US Dollar", "+691", "Oceania", 115_000),
    "moldova": ("Chisinau", "MD", "MDA", "MDL", "Leu", "+373", "Europe", 2_600_000),
    "monaco": ("Monaco", "MC", "MCO", "EUR", "Euro", "+377", "Europe", 36_000),
    "mongolia": ("Ulaanbaatar", "MN", "MNG", "MNT", "Tugrik", "+976", "Asia", 3_398_000),
    "montenegro": ("Podgorica", "ME", "MNE", "EUR", "Euro", "+382", "Europe", 620_000),
    "morocco": ("Rabat", "MA", "MAR", "MAD", "Dirham", "+212", "Africa", 37_840_000),
    "mozambique": ("Maputo", "MZ", "MOZ", "MZN", "Metical", "+258", "Africa", 33_897_000),
    "myanmar": ("Naypyidaw", "MM", "MMR", "MMK", "Kyat", "+95", "Asia", 54_179_000),
    "namibia": ("Windhoek", "NA", "NAM", "NAD", "Namibian Dollar", "+264", "Africa", 2_604_000),
    "nauru": ("Yaren", "NR", "NRU", "AUD", "Australian Dollar", "+674", "Oceania", 13_000),
    "nepal": ("Kathmandu", "NP", "NPL", "NPR", "Nepalese Rupee", "+977", "Asia", 30_896_000),
    "netherlands": ("Amsterdam", "NL", "NLD", "EUR", "Euro", "+31", "Europe", 17_618_000),
    "new zealand": ("Wellington", "NZ", "NZL", "NZD", "New Zealand Dollar", "+64", "Oceania", 5_186_000),
    "nicaragua": ("Managua", "NI", "NIC", "NIO", "Cordoba", "+505", "Americas", 7_046_000),
    "niger": ("Niamey", "NE", "NER", "XOF", "CFA Franc", "+227", "Africa", 27_202_000),
    "nigeria": ("Abuja", "NG", "NGA", "NGN", "Naira", "+234", "Africa", 223_804_000),
    "north korea": ("Pyongyang", "KP", "PRK", "KPW", "Won", "+850", "Asia", 26_161_000),
    "north macedonia": ("Skopje", "MK", "MKD", "MKD", "Denar", "+389", "Europe", 1_836_000),
    "norway": ("Oslo", "NO", "NOR", "NOK", "Krone", "+47", "Europe", 5_474_000),
    "oman": ("Muscat", "OM", "OMN", "OMR", "Rial", "+968", "Asia", 4_644_000),
    "pakistan": ("Islamabad", "PK", "PAK", "PKR", "Pakistani Rupee", "+92", "Asia", 240_486_000),
    "palau": ("Ngerulmud", "PW", "PLW", "USD", "US Dollar", "+680", "Oceania", 18_000),
    "palestine": ("Ramallah", "PS", "PSE", "ILS", "Shekel", "+970", "Asia", 5_483_000),
    "panama": ("Panama City", "PA", "PAN", "PAB", "Balboa", "+507", "Americas", 4_468_000),
    "papua new guinea": ("Port Moresby", "PG", "PNG", "PGK", "Kina", "+675", "Oceania", 10_143_000),
    "paraguay": ("Asuncion", "PY", "PRY", "PYG", "Guarani", "+595", "Americas", 6_862_000),
    "peru": ("Lima", "PE", "PER", "PEN", "Sol", "+51", "Americas", 34_352_000),
    "philippines": ("Manila", "PH", "PHL", "PHP", "Philippine Peso", "+63", "Asia", 117_337_000),
    "poland": ("Warsaw", "PL", "POL", "PLN", "Zloty", "+48", "Europe", 36_753_000),
    "portugal": ("Lisbon", "PT", "PRT", "EUR", "Euro", "+351", "Europe", 10_348_000),
    "qatar": ("Doha", "QA", "QAT", "QAR", "Riyal", "+974", "Asia", 2_696_000),
    "romania": ("Bucharest", "RO", "ROU", "RON", "Leu", "+40", "Europe", 19_038_000),
    "russia": ("Moscow", "RU", "RUS", "RUB", "Ruble", "+7", "Europe", 144_236_000),
    "rwanda": ("Kigali", "RW", "RWA", "RWF", "Rwandan Franc", "+250", "Africa", 14_094_000),
    "saint kitts and nevis": ("Basseterre", "KN", "KNA", "XCD", "East Caribbean Dollar", "+1-869", "Americas", 48_000),
    "saint lucia": ("Castries", "LC", "LCA", "XCD", "East Caribbean Dollar", "+1-758", "Americas", 180_000),
    "saint vincent and the grenadines": ("Kingstown", "VC", "VCT", "XCD", "East Caribbean Dollar", "+1-784", "Americas", 104_000),
    "samoa": ("Apia", "WS", "WSM", "WST", "Tala", "+685", "Oceania", 222_000),
    "san marino": ("San Marino", "SM", "SMR", "EUR", "Euro", "+378", "Europe", 33_000),
    "sao tome and principe": ("Sao Tome", "ST", "STP", "STN", "Dobra", "+239", "Africa", 228_000),
    "saudi arabia": ("Riyadh", "SA", "SAU", "SAR", "Riyal", "+966", "Asia", 36_948_000),
    "senegal": ("Dakar", "SN", "SEN", "XOF", "CFA Franc", "+221", "Africa", 17_763_000),
    "serbia": ("Belgrade", "RS", "SRB", "RSD", "Serbian Dinar", "+381", "Europe", 6_664_000),
    "seychelles": ("Victoria", "SC", "SYC", "SCR", "Seychellois Rupee", "+248", "Africa", 120_000),
    "sierra leone": ("Freetown", "SL", "SLE", "SLL", "Leone", "+232", "Africa", 8_606_000),
    "singapore": ("Singapore", "SG", "SGP", "SGD", "Singapore Dollar", "+65", "Asia", 5_917_000),
    "slovakia": ("Bratislava", "SK", "SVK", "EUR", "Euro", "+421", "Europe", 5_460_000),
    "slovenia": ("Ljubljana", "SI", "SVN", "EUR", "Euro", "+386", "Europe", 2_120_000),
    "solomon islands": ("Honiara", "SB", "SLB", "SBD", "Solomon Islands Dollar", "+677", "Oceania", 724_000),
    "somalia": ("Mogadishu", "SO", "SOM", "SOS", "Somali Shilling", "+252", "Africa", 18_143_000),
    "south africa": ("Pretoria", "ZA", "ZAF", "ZAR", "Rand", "+27", "Africa", 60_414_000),
    "south korea": ("Seoul", "KR", "KOR", "KRW", "Won", "+82", "Asia", 51_745_000),
    "south sudan": ("Juba", "SS", "SSD", "SSP", "South Sudanese Pound", "+211", "Africa", 11_088_000),
    "spain": ("Madrid", "ES", "ESP", "EUR", "Euro", "+34", "Europe", 47_520_000),
    "sri lanka": ("Sri Jayawardenepura Kotte", "LK", "LKA", "LKR", "Sri Lankan Rupee", "+94", "Asia", 22_037_000),
    "sudan": ("Khartoum", "SD", "SDN", "SDG", "Sudanese Pound", "+249", "Africa", 48_109_000),
    "suriname": ("Paramaribo", "SR", "SUR", "SRD", "Surinamese Dollar", "+597", "Americas", 618_000),
    "sweden": ("Stockholm", "SE", "SWE", "SEK", "Krona", "+46", "Europe", 10_522_000),
    "switzerland": ("Bern", "CH", "CHE", "CHF", "Swiss Franc", "+41", "Europe", 8_796_000),
    "syria": ("Damascus", "SY", "SYR", "SYP", "Syrian Pound", "+963", "Asia", 22_125_000),
    "taiwan": ("Taipei", "TW", "TWN", "TWD", "New Taiwan Dollar", "+886", "Asia", 23_894_000),
    "tajikistan": ("Dushanbe", "TJ", "TJK", "TJS", "Somoni", "+992", "Asia", 10_143_000),
    "tanzania": ("Dodoma", "TZ", "TZA", "TZS", "Tanzanian Shilling", "+255", "Africa", 65_498_000),
    "thailand": ("Bangkok", "TH", "THA", "THB", "Baht", "+66", "Asia", 71_801_000),
    "timor-leste": ("Dili", "TL", "TLS", "USD", "US Dollar", "+670", "Asia", 1_341_000),
    "togo": ("Lome", "TG", "TGO", "XOF", "CFA Franc", "+228", "Africa", 8_849_000),
    "tonga": ("Nuku'alofa", "TO", "TON", "TOP", "Pa'anga", "+676", "Oceania", 107_000),
    "trinidad and tobago": ("Port of Spain", "TT", "TTO", "TTD", "Trinidad Dollar", "+1-868", "Americas", 1_534_000),
    "tunisia": ("Tunis", "TN", "TUN", "TND", "Tunisian Dinar", "+216", "Africa", 12_458_000),
    "turkey": ("Ankara", "TR", "TUR", "TRY", "Lira", "+90", "Asia", 85_326_000),
    "turkmenistan": ("Ashgabat", "TM", "TKM", "TMT", "Manat", "+993", "Asia", 6_431_000),
    "tuvalu": ("Funafuti", "TV", "TUV", "AUD", "Australian Dollar", "+688", "Oceania", 11_000),
    "uganda": ("Kampala", "UG", "UGA", "UGX", "Ugandan Shilling", "+256", "Africa", 48_583_000),
    "ukraine": ("Kyiv", "UA", "UKR", "UAH", "Hryvnia", "+380", "Europe", 37_000_000),
    "united arab emirates": ("Abu Dhabi", "AE", "ARE", "AED", "Dirham", "+971", "Asia", 9_441_000),
    "united kingdom": ("London", "GB", "GBR", "GBP", "Pound Sterling", "+44", "Europe", 67_736_000),
    "united states": ("Washington, D.C.", "US", "USA", "USD", "US Dollar", "+1", "Americas", 339_997_000),
    "uruguay": ("Montevideo", "UY", "URY", "UYU", "Uruguayan Peso", "+598", "Americas", 3_423_000),
    "uzbekistan": ("Tashkent", "UZ", "UZB", "UZS", "Som", "+998", "Asia", 35_163_000),
    "vanuatu": ("Port Vila", "VU", "VUT", "VUV", "Vatu", "+678", "Oceania", 326_000),
    "vatican city": ("Vatican City", "VA", "VAT", "EUR", "Euro", "+39-06", "Europe", 800),
    "venezuela": ("Caracas", "VE", "VEN", "VES", "Bolivar", "+58", "Americas", 28_838_000),
    "vietnam": ("Hanoi", "VN", "VNM", "VND", "Dong", "+84", "Asia", 98_858_000),
    "yemen": ("Sanaa", "YE", "YEM", "YER", "Yemeni Rial", "+967", "Asia", 34_449_000),
    "zambia": ("Lusaka", "ZM", "ZMB", "ZMW", "Kwacha", "+260", "Africa", 20_569_000),
    "zimbabwe": ("Harare", "ZW", "ZWE", "ZWL", "Zimbabwe Dollar", "+263", "Africa", 16_665_000),
}

# Aliases for common alternate names
_ALIASES = {
    "usa": "united states", "us": "united states", "america": "united states",
    "uk": "united kingdom", "britain": "united kingdom", "great britain": "united kingdom", "england": "united kingdom",
    "uae": "united arab emirates", "emirates": "united arab emirates",
    "south korea": "south korea", "korea": "south korea", "rok": "south korea",
    "north korea": "north korea", "dprk": "north korea",
    "russia": "russia", "russian federation": "russia",
    "iran": "iran", "persia": "iran",
    "ivory coast": "ivory coast", "cote d'ivoire": "ivory coast",
    "congo dr": "congo dr", "drc": "congo dr", "democratic republic of the congo": "congo dr",
    "congo": "congo", "republic of the congo": "congo",
    "czech republic": "czechia", "czech": "czechia",
    "burma": "myanmar",
    "holland": "netherlands",
    "cape verde": "cabo verde",
    "swaziland": "eswatini",
    "east timor": "timor-leste",
    "vatican": "vatican city", "holy see": "vatican city",
}


def _lookup(name: str) -> tuple | None:
    """Normalize and look up country data."""
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    return _COUNTRIES.get(key)


def capital(country: str) -> str:
    """Capital city of a country."""
    data = _lookup(country)
    return data[0] if data else f"unknown country: {country}"


def country_iso2(country: str) -> str:
    """ISO 3166-1 alpha-2 code."""
    data = _lookup(country)
    return data[1] if data else f"unknown country: {country}"


def country_iso3(country: str) -> str:
    """ISO 3166-1 alpha-3 code."""
    data = _lookup(country)
    return data[2] if data else f"unknown country: {country}"


def country_currency(country: str) -> str:
    """Currency code and name."""
    data = _lookup(country)
    return f"{data[3]} ({data[4]})" if data else f"unknown country: {country}"


def country_calling_code(country: str) -> str:
    """International calling code."""
    data = _lookup(country)
    return data[5] if data else f"unknown country: {country}"


def country_region(country: str) -> str:
    """World region (Africa, Americas, Asia, Europe, Oceania)."""
    data = _lookup(country)
    return data[6] if data else f"unknown country: {country}"


def country_population(country: str) -> str:
    """Approximate population (as of 2025)."""
    data = _lookup(country)
    if not data:
        return f"unknown country: {country}"
    pop = data[7]
    if pop >= 1_000_000_000:
        return f"~{pop / 1_000_000_000:.2f} billion"
    if pop >= 1_000_000:
        return f"~{pop / 1_000_000:.1f} million"
    return f"~{pop:,}"


def country_info(country: str) -> str:
    """Full info summary for a country."""
    data = _lookup(country)
    if not data:
        return f"unknown country: {country}"
    cap, iso2, iso3, curr, curr_name, call, region, pop = data
    return (f"Capital: {cap}, ISO: {iso2}/{iso3}, Currency: {curr} ({curr_name}), "
            f"Calling: {call}, Region: {region}, Pop: ~{pop:,}")


COUNTRY_FUNCTIONS = {
    "capital": capital,
    "country_iso2": country_iso2,
    "country_iso3": country_iso3,
    "country_currency": country_currency,
    "country_calling_code": country_calling_code,
    "country_region": country_region,
    "country_population": country_population,
    "country_info": country_info,
}
