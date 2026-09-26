"""Watch used-bike marketplaces for Kyoot bikes and push new matches to ntfy."""
import json
import os
import re
import sys
import time
import urllib.parse
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "kyoot-688cb1ec83f1")
MAX_PRICE = 1600  # alert only if price < this (unknown price still alerts)
STATE_FILE = "state.json"
FAIL_ALERT_AFTER = 3  # consecutive failed runs before a "source broken" alert
DRY_RUN = os.environ.get("DRY_RUN") == "1"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")
S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})

ASK_TO_SHIP = ("Hi! Is your Kyoot still available? I'm in the SF Bay Area and would "
               "cover shipping (I can send a prepaid BikeFlights label and box). "
               "Would you be open to shipping it?")

KYOOT = re.compile(r"\bkyoot", re.I)
ROLLY = re.compile(r"\brol{1,2}y[\s-]*pol{1,2}y\b", re.I)
EXCLUDE = re.compile(r"rivendell|\btires?\b|\btyres?\b|\btoy\b", re.I)
WANTED = re.compile(r"^\s*(wtb|iso|wanted|looking for)\b", re.I)
PICKUP = re.compile(r"pick[\s-]?up only|local pick[\s-]?up|no shipping|"
                    r"will not ship|won'?t ship|not shipping|local only", re.I)


def is_match(title, desc=""):
    title, desc = title or "", desc or ""
    if WANTED.search(title):
        return False
    if KYOOT.search(title) or KYOOT.search(desc):
        return True
    return bool(ROLLY.search(title) and not EXCLUDE.search(title))


def parse_price(text):
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    m = re.search(r"\$\s?([\d,]+(?:\.\d\d)?)", str(text))
    return float(m.group(1).replace(",", "")) if m else None


def get(url, **kw):
    r = S.get(url, timeout=30, **kw)
    r.raise_for_status()
    return r


def item(source, id_, title, url, price=None, location="", pickup=False):
    return dict(key=f"{source}:{id_}", source=source, title=(title or "").strip(), url=url,
                price=price, location=location or "", pickup=pickup)


# ---------------------------------------------------------------- sources

def rad_bazaar():
    """The Radavist Rad Bazaar: public Algolia search index used by the site."""
    out = []
    for q in ("kyoot", "rolly polly", "roly poly"):
        r = S.post("https://rmp8php3mt-dsn.algolia.net/1/indexes/listings_created_desc/query",
                   headers={"X-Algolia-Application-Id": "RMP8PHP3MT",
                            "X-Algolia-API-Key": "f341165624a0f1699ce24f259e8e4b81"},
                   json={"query": q, "hitsPerPage": 50}, timeout=30)
        r.raise_for_status()
        for h in r.json()["hits"]:
            if h.get("status") != "active":
                continue
            out.append(item("radbazaar", h["objectID"], h.get("title", ""),
                            f"https://theradavist.com/bazaar/item/{h['objectID']}",
                            price=(h["price"] / 100) if h.get("price") else None,
                            location=(h.get("location") or {}).get("description", ""),
                            pickup=h.get("delivery") == "pickup")
                       | {"desc": h.get("description", "")})
    return out


def bikepacking_coop():
    """BIKEPACKING.com Bike Camp Co-op: RSS feed of newest listings."""
    soup = BeautifulSoup(get("https://bikepacking.com/basecamp/bike-camp-co-op/feed/").content, "xml")
    out = []
    for it in soup.find_all("item"):
        link = it.link.text.strip()
        desc = BeautifulSoup(it.description.text if it.description else "", "html.parser").get_text(" ")
        out.append(item("bikepacking", link, it.title.text, link,
                        price=parse_price(desc), pickup=bool(PICKUP.search(desc))) | {"desc": desc})
    return out


def craigslist():
    """Every US Craigslist site, all for-sale categories."""
    areas = get("https://reference.craigslist.org/Areas").json()
    hosts = sorted({a["Hostname"] for a in areas if a.get("Country") == "US"})
    q = urllib.parse.quote('kyoot|"rolly polly"|"roly poly"')
    out, errors = [], 0
    for h in hosts:
        try:
            html = get(f"https://{h}.craigslist.org/search/sss?query={q}").text
        except Exception:
            errors += 1
            continue
        for li in BeautifulSoup(html, "html.parser").select("li.cl-static-search-result"):
            a = li.find("a")
            if not a:
                continue
            title = (li.select_one(".title") or a).get_text(" ", strip=True)
            price = li.select_one(".price")
            loc = li.select_one(".location")
            url = a["href"]
            out.append(item("craigslist", url.split("?")[0], title, url,
                            price=parse_price(price.get_text() if price else None),
                            location=(loc.get_text(strip=True) if loc else h), pickup=True))
        time.sleep(0.4)
    if errors > len(hosts) / 2:
        raise RuntimeError(f"craigslist: {errors}/{len(hosts)} sites failed")
    return out


def ebay():
    out = []
    html = get("https://www.ebay.com/sch/i.html?_nkw=kyoot&_sop=10&_ipg=120").text
    soup = BeautifulSoup(html, "html.parser")
    for card in soup.select("li.s-item, li.s-card"):
        a = card.select_one("a[href*='/itm/']")
        t = card.select_one(".s-item__title, .s-card__title")
        if not a or not t:
            continue
        m = re.search(r"/itm/(\d+)", a["href"])
        if not m:
            continue
        p = card.select_one(".s-item__price, .s-card__price")
        text = card.get_text(" ")
        out.append(item("ebay", m.group(1), t.get_text(" ", strip=True),
                        f"https://www.ebay.com/itm/{m.group(1)}",
                        price=parse_price(p.get_text() if p else None),
                        pickup=bool(re.search(r"local pickup|pickup only", text, re.I)
                                    and not re.search(r"shipping", text, re.I))))
    if not out and "captcha" in html.lower():
        raise RuntimeError("ebay: blocked by captcha")
    return out


def pinkbike():
    out = []
    html = get("https://www.pinkbike.com/buysell/list/?q=kyoot").text
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select("a[href*='/buysell/']"):
        m = re.search(r"/buysell/(\d+)/?", a["href"])
        title = a.get_text(" ", strip=True)
        if not m or not title:
            continue
        box = a.find_parent("div")
        text = box.get_text(" ") if box else ""
        out.append(item("pinkbike", m.group(1), title, f"https://www.pinkbike.com/buysell/{m.group(1)}/",
                        price=parse_price(text), pickup=bool(PICKUP.search(text))))
    return out


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def offerup():
    html = get("https://offerup.com/search?q=kyoot").text
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise RuntimeError("offerup: page data not found (layout changed or blocked)")
    out = []
    for d in _walk(json.loads(m.group(1))):
        lid, title = d.get("listingId"), d.get("title")
        if lid and isinstance(title, str):
            out.append(item("offerup", lid, title, f"https://offerup.com/item/detail/{lid}",
                            price=parse_price(d.get("price")),
                            location=d.get("locationName", "")))
    return out


def buycycle():
    html = get("https://buycycle.com/en-us/shop/search/kyoot").text
    out = []
    for a in BeautifulSoup(html, "html.parser").select("a[href*='/product/']"):
        title = a.get_text(" ", strip=True)
        href = urllib.parse.urljoin("https://buycycle.com", a["href"])
        out.append(item("buycycle", href.split("?")[0], title[:120], href, price=parse_price(title)))
    return out


SOURCES = [rad_bazaar, bikepacking_coop, ebay, pinkbike, offerup, buycycle, craigslist]


# ---------------------------------------------------------------- notify

def notify(title, body, url=None, tags="bike"):
    print(f"NOTIFY: {title} | {body} | {url}")
    if DRY_RUN:
        return
    headers = {"Title": title.encode("utf-8"), "Tags": tags}
    if url:
        headers["Click"] = url
    requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=body.encode("utf-8"),
                  headers=headers, timeout=30)


def main():
    state = {"seen": {}, "fails": {}}
    if os.path.exists(STATE_FILE):
        state.update(json.load(open(STATE_FILE)))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = []

    for src in SOURCES:
        name = src.__name__
        try:
            items = src()
            matches = [i for i in items if is_match(i["title"], i.get("desc", ""))
                       and (i["price"] is None or i["price"] < MAX_PRICE)]
        except Exception as e:
            n = state["fails"].get(name, 0) + 1
            state["fails"][name] = n
            summary.append(f"{name}: FAILED ({n}x) {e}")
            if n == FAIL_ALERT_AFTER:
                notify(f"Kyoot watch: {name} is failing",
                       f"{name} has failed {n} runs in a row: {e}", tags="warning")
            continue
        if state["fails"].get(name, 0) >= FAIL_ALERT_AFTER:
            notify(f"Kyoot watch: {name} is working again", "Source recovered.", tags="white_check_mark")
        state["fails"][name] = 0

        new = [i for i in matches if i["key"] not in state["seen"]]
        summary.append(f"{name}: {len(items)} scanned, {len(matches)} match, {len(new)} new")
        for i in new:
            state["seen"][i["key"]] = now
            price = f"${i['price']:,.0f}" if i["price"] is not None else "price ?"
            body = f"{price} · {i['location'] or 'location ?'} · {name}"
            if i["pickup"]:
                body += f"\nPICKUP ONLY — ask to ship:\n{ASK_TO_SHIP}"
            try:
                notify(i["title"][:120], body, i["url"])
            except Exception as e:
                summary.append(f"notify failed: {e}")

    state["last_run"] = now  # daily commit keeps GitHub from pausing the schedule
    state["last_summary"] = summary
    print("\n".join(summary))
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    sys.exit(main())
