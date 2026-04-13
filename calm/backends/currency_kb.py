"""
CALM Currency knowledge backend — ISO 4217 codes, symbols, decimal places.

Models confuse currency symbols, mix up ISO codes, hallucinate decimals.
Covers all actively-traded currencies + major crypto.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

# (name, symbol, decimal_places, country_or_region)
_CURRENCIES = {
    "AED": ("UAE Dirham", "د.إ", 2, "United Arab Emirates"),
    "AFN": ("Afghani", "؋", 2, "Afghanistan"),
    "ALL": ("Lek", "L", 2, "Albania"),
    "AMD": ("Armenian Dram", "֏", 2, "Armenia"),
    "ANG": ("Netherlands Antillean Guilder", "ƒ", 2, "Curacao, Sint Maarten"),
    "AOA": ("Kwanza", "Kz", 2, "Angola"),
    "ARS": ("Argentine Peso", "$", 2, "Argentina"),
    "AUD": ("Australian Dollar", "A$", 2, "Australia"),
    "AWG": ("Aruban Florin", "ƒ", 2, "Aruba"),
    "AZN": ("Azerbaijani Manat", "₼", 2, "Azerbaijan"),
    "BAM": ("Convertible Mark", "KM", 2, "Bosnia and Herzegovina"),
    "BBD": ("Barbados Dollar", "Bds$", 2, "Barbados"),
    "BDT": ("Taka", "৳", 2, "Bangladesh"),
    "BGN": ("Bulgarian Lev", "лв", 2, "Bulgaria"),
    "BHD": ("Bahraini Dinar", ".د.ب", 3, "Bahrain"),
    "BIF": ("Burundian Franc", "FBu", 0, "Burundi"),
    "BMD": ("Bermudian Dollar", "$", 2, "Bermuda"),
    "BND": ("Brunei Dollar", "B$", 2, "Brunei"),
    "BOB": ("Boliviano", "Bs.", 2, "Bolivia"),
    "BRL": ("Brazilian Real", "R$", 2, "Brazil"),
    "BSD": ("Bahamian Dollar", "$", 2, "Bahamas"),
    "BTC": ("Bitcoin", "₿", 8, "Cryptocurrency"),
    "BTN": ("Ngultrum", "Nu.", 2, "Bhutan"),
    "BWP": ("Pula", "P", 2, "Botswana"),
    "BYN": ("Belarusian Ruble", "Br", 2, "Belarus"),
    "BZD": ("Belize Dollar", "BZ$", 2, "Belize"),
    "CAD": ("Canadian Dollar", "C$", 2, "Canada"),
    "CDF": ("Congolese Franc", "FC", 2, "DR Congo"),
    "CHF": ("Swiss Franc", "CHF", 2, "Switzerland"),
    "CLP": ("Chilean Peso", "$", 0, "Chile"),
    "CNY": ("Yuan Renminbi", "¥", 2, "China"),
    "COP": ("Colombian Peso", "$", 2, "Colombia"),
    "CRC": ("Costa Rican Colon", "₡", 2, "Costa Rica"),
    "CUP": ("Cuban Peso", "₱", 2, "Cuba"),
    "CVE": ("Cape Verdean Escudo", "$", 2, "Cabo Verde"),
    "CZK": ("Czech Koruna", "Kč", 2, "Czechia"),
    "DJF": ("Djiboutian Franc", "Fdj", 0, "Djibouti"),
    "DKK": ("Danish Krone", "kr", 2, "Denmark"),
    "DOP": ("Dominican Peso", "RD$", 2, "Dominican Republic"),
    "DZD": ("Algerian Dinar", "د.ج", 2, "Algeria"),
    "EGP": ("Egyptian Pound", "E£", 2, "Egypt"),
    "ETB": ("Ethiopian Birr", "Br", 2, "Ethiopia"),
    "ETH": ("Ether", "Ξ", 18, "Cryptocurrency"),
    "EUR": ("Euro", "€", 2, "Eurozone"),
    "FJD": ("Fiji Dollar", "FJ$", 2, "Fiji"),
    "GBP": ("Pound Sterling", "£", 2, "United Kingdom"),
    "GEL": ("Georgian Lari", "₾", 2, "Georgia"),
    "GHS": ("Ghana Cedi", "GH₵", 2, "Ghana"),
    "GMD": ("Dalasi", "D", 2, "Gambia"),
    "GNF": ("Guinean Franc", "FG", 0, "Guinea"),
    "GTQ": ("Quetzal", "Q", 2, "Guatemala"),
    "GYD": ("Guyana Dollar", "G$", 2, "Guyana"),
    "HKD": ("Hong Kong Dollar", "HK$", 2, "Hong Kong"),
    "HNL": ("Lempira", "L", 2, "Honduras"),
    "HTG": ("Gourde", "G", 2, "Haiti"),
    "HUF": ("Forint", "Ft", 2, "Hungary"),
    "IDR": ("Rupiah", "Rp", 2, "Indonesia"),
    "ILS": ("New Israeli Shekel", "₪", 2, "Israel"),
    "INR": ("Indian Rupee", "₹", 2, "India"),
    "IQD": ("Iraqi Dinar", "ع.د", 3, "Iraq"),
    "IRR": ("Iranian Rial", "﷼", 2, "Iran"),
    "ISK": ("Icelandic Krona", "kr", 0, "Iceland"),
    "JMD": ("Jamaican Dollar", "J$", 2, "Jamaica"),
    "JOD": ("Jordanian Dinar", "JD", 3, "Jordan"),
    "JPY": ("Yen", "¥", 0, "Japan"),
    "KES": ("Kenyan Shilling", "KSh", 2, "Kenya"),
    "KGS": ("Kyrgyzstani Som", "сом", 2, "Kyrgyzstan"),
    "KHR": ("Riel", "៛", 2, "Cambodia"),
    "KMF": ("Comorian Franc", "CF", 0, "Comoros"),
    "KRW": ("South Korean Won", "₩", 0, "South Korea"),
    "KWD": ("Kuwaiti Dinar", "د.ك", 3, "Kuwait"),
    "KYD": ("Cayman Islands Dollar", "CI$", 2, "Cayman Islands"),
    "KZT": ("Tenge", "₸", 2, "Kazakhstan"),
    "LAK": ("Lao Kip", "₭", 2, "Laos"),
    "LBP": ("Lebanese Pound", "ل.ل", 2, "Lebanon"),
    "LKR": ("Sri Lankan Rupee", "Rs", 2, "Sri Lanka"),
    "LRD": ("Liberian Dollar", "L$", 2, "Liberia"),
    "LSL": ("Loti", "L", 2, "Lesotho"),
    "LYD": ("Libyan Dinar", "ل.د", 3, "Libya"),
    "MAD": ("Moroccan Dirham", "د.م.", 2, "Morocco"),
    "MDL": ("Moldovan Leu", "L", 2, "Moldova"),
    "MGA": ("Malagasy Ariary", "Ar", 2, "Madagascar"),
    "MKD": ("Denar", "ден", 2, "North Macedonia"),
    "MMK": ("Kyat", "K", 2, "Myanmar"),
    "MNT": ("Tugrik", "₮", 2, "Mongolia"),
    "MOP": ("Pataca", "MOP$", 2, "Macau"),
    "MRU": ("Ouguiya", "UM", 2, "Mauritania"),
    "MUR": ("Mauritian Rupee", "₨", 2, "Mauritius"),
    "MVR": ("Rufiyaa", "Rf", 2, "Maldives"),
    "MWK": ("Malawian Kwacha", "MK", 2, "Malawi"),
    "MXN": ("Mexican Peso", "Mex$", 2, "Mexico"),
    "MYR": ("Malaysian Ringgit", "RM", 2, "Malaysia"),
    "MZN": ("Mozambican Metical", "MT", 2, "Mozambique"),
    "NAD": ("Namibian Dollar", "N$", 2, "Namibia"),
    "NGN": ("Naira", "₦", 2, "Nigeria"),
    "NIO": ("Cordoba Oro", "C$", 2, "Nicaragua"),
    "NOK": ("Norwegian Krone", "kr", 2, "Norway"),
    "NPR": ("Nepalese Rupee", "₨", 2, "Nepal"),
    "NZD": ("New Zealand Dollar", "NZ$", 2, "New Zealand"),
    "OMR": ("Rial Omani", "ر.ع.", 3, "Oman"),
    "PAB": ("Balboa", "B/.", 2, "Panama"),
    "PEN": ("Sol", "S/.", 2, "Peru"),
    "PGK": ("Kina", "K", 2, "Papua New Guinea"),
    "PHP": ("Philippine Peso", "₱", 2, "Philippines"),
    "PKR": ("Pakistan Rupee", "₨", 2, "Pakistan"),
    "PLN": ("Zloty", "zł", 2, "Poland"),
    "PYG": ("Guarani", "₲", 0, "Paraguay"),
    "QAR": ("Qatari Riyal", "ر.ق", 2, "Qatar"),
    "RON": ("Romanian Leu", "lei", 2, "Romania"),
    "RSD": ("Serbian Dinar", "din.", 2, "Serbia"),
    "RUB": ("Russian Ruble", "₽", 2, "Russia"),
    "RWF": ("Rwanda Franc", "RF", 0, "Rwanda"),
    "SAR": ("Saudi Riyal", "﷼", 2, "Saudi Arabia"),
    "SBD": ("Solomon Islands Dollar", "SI$", 2, "Solomon Islands"),
    "SCR": ("Seychellois Rupee", "₨", 2, "Seychelles"),
    "SDG": ("Sudanese Pound", "ج.س.", 2, "Sudan"),
    "SEK": ("Swedish Krona", "kr", 2, "Sweden"),
    "SGD": ("Singapore Dollar", "S$", 2, "Singapore"),
    "SLL": ("Leone", "Le", 2, "Sierra Leone"),
    "SOS": ("Somali Shilling", "Sh", 2, "Somalia"),
    "SRD": ("Surinamese Dollar", "SRD", 2, "Suriname"),
    "SSP": ("South Sudanese Pound", "£", 2, "South Sudan"),
    "STN": ("Dobra", "Db", 2, "Sao Tome and Principe"),
    "SYP": ("Syrian Pound", "£S", 2, "Syria"),
    "SZL": ("Lilangeni", "E", 2, "Eswatini"),
    "THB": ("Thai Baht", "฿", 2, "Thailand"),
    "TJS": ("Somoni", "SM", 2, "Tajikistan"),
    "TMT": ("Turkmenistani Manat", "T", 2, "Turkmenistan"),
    "TND": ("Tunisian Dinar", "د.ت", 3, "Tunisia"),
    "TOP": ("Pa'anga", "T$", 2, "Tonga"),
    "TRY": ("Turkish Lira", "₺", 2, "Turkey"),
    "TTD": ("Trinidad and Tobago Dollar", "TT$", 2, "Trinidad and Tobago"),
    "TWD": ("New Taiwan Dollar", "NT$", 2, "Taiwan"),
    "TZS": ("Tanzanian Shilling", "TSh", 2, "Tanzania"),
    "UAH": ("Hryvnia", "₴", 2, "Ukraine"),
    "UGX": ("Uganda Shilling", "USh", 0, "Uganda"),
    "USD": ("US Dollar", "$", 2, "United States"),
    "USDT": ("Tether", "₮", 6, "Cryptocurrency"),
    "UYU": ("Peso Uruguayo", "$U", 2, "Uruguay"),
    "UZS": ("Uzbekistani Sum", "сўм", 2, "Uzbekistan"),
    "VES": ("Bolivar Soberano", "Bs.S", 2, "Venezuela"),
    "VND": ("Dong", "₫", 0, "Vietnam"),
    "VUV": ("Vatu", "VT", 0, "Vanuatu"),
    "WST": ("Tala", "WS$", 2, "Samoa"),
    "XAF": ("CFA Franc BEAC", "FCFA", 0, "Central Africa"),
    "XCD": ("East Caribbean Dollar", "EC$", 2, "Eastern Caribbean"),
    "XOF": ("CFA Franc BCEAO", "CFA", 0, "West Africa"),
    "XPF": ("CFP Franc", "₣", 0, "French Pacific"),
    "YER": ("Yemeni Rial", "﷼", 2, "Yemen"),
    "ZAR": ("South African Rand", "R", 2, "South Africa"),
    "ZMW": ("Zambian Kwacha", "ZK", 2, "Zambia"),
    "ZWL": ("Zimbabwe Dollar", "Z$", 2, "Zimbabwe"),
}


def currency_name(code: str) -> str:
    """Full name of a currency by ISO 4217 code."""
    entry = _CURRENCIES.get(str(code).upper())
    return entry[0] if entry else f"Unknown currency: {code}"


def currency_symbol(code: str) -> str:
    """Symbol for a currency by ISO 4217 code."""
    entry = _CURRENCIES.get(str(code).upper())
    return entry[1] if entry else "?"


def currency_decimals(code: str) -> int:
    """Number of decimal places for a currency (e.g. JPY=0, USD=2, BHD=3)."""
    entry = _CURRENCIES.get(str(code).upper())
    return entry[2] if entry else -1


def currency_country(code: str) -> str:
    """Primary country/region for a currency code."""
    entry = _CURRENCIES.get(str(code).upper())
    return entry[3] if entry else f"Unknown currency: {code}"


def is_zero_decimal(code: str) -> bool:
    """Whether a currency uses zero decimal places (e.g. JPY, KRW)."""
    entry = _CURRENCIES.get(str(code).upper())
    return entry[2] == 0 if entry else False


def is_three_decimal(code: str) -> bool:
    """Whether a currency uses three decimal places (e.g. BHD, KWD, OMR)."""
    entry = _CURRENCIES.get(str(code).upper())
    return entry[2] == 3 if entry else False


def list_by_region(region: str) -> list[str]:
    """List all currency codes for a given region/country."""
    r = str(region).lower()
    return sorted(code for code, entry in _CURRENCIES.items()
                  if r in entry[3].lower())


def list_zero_decimal() -> list[str]:
    """List all zero-decimal currencies."""
    return sorted(code for code, entry in _CURRENCIES.items() if entry[2] == 0)


CURRENCY_FUNCTIONS = {
    "currency_name": currency_name,
    "currency_symbol": currency_symbol,
    "currency_decimals": currency_decimals,
    "currency_country": currency_country,
    "is_zero_decimal": is_zero_decimal,
    "is_three_decimal": is_three_decimal,
    "list_by_region": list_by_region,
    "list_zero_decimal": list_zero_decimal,
}

CURRENCY_NL_PATTERNS = [
    (r'(?:what is|what\'s) the (?:currency|money) (?:of|in|used in) (\w[\w\s]+)', None),
    (r'(?:symbol|sign) (?:for|of) (\w{3})\b', 'currency_symbol("{0}")'),
    (r'(?:how many|number of) decimal (?:places?|digits?) (?:for|in|does) (\w{3})', 'currency_decimals("{0}")'),
    (r'(?:is) (\w{3}) (?:a )?zero.decimal', 'is_zero_decimal("{0}")'),
]
