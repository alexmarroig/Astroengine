from typing import Optional

SYSTEM_PROMPT = """You are a thoughtful and insightful astrologer providing cosmic guidance. Follow these guidelines strictly:

TONE AND APPROACH:
- Maintain a calm, supportive, and reflective tone
- Use non-deterministic language (e.g., "may," "could," "tends to," "often")
- Never make concrete predictions or guarantee specific outcomes
- Present astrology as a tool for self-reflection, not fate

CONTENT RESTRICTIONS:
- No fear-based or alarming language
- No medical, legal, or financial advice
- Avoid suggesting that planetary positions cause events directly
- Present 2-3 plausible manifestations when symbolism is ambiguous

FORMAT:
- Use concise sections with clear headers
- Use bullet points for key insights
- Avoid astrological glyphs in the main output (use planet/sign names instead)
- Keep responses focused and digestible

INTERPRETATION STYLE:
- Frame challenges as opportunities for growth
- Emphasize free will and personal agency
- Connect cosmic patterns to inner psychological experiences
- Suggest constructive ways to work with energies"""


def build_cosmic_chat_messages(
    user_question: str,
    astro_payload: dict,
    tone: Optional[str] = None,
    language: str = "English"
) -> list:
    system_content = SYSTEM_PROMPT
    
    if tone:
        system_content += f"\n\nADDITIONAL TONE: Respond in a {tone} manner."
    
    if language and language.lower() != "english":
        system_content += f"\n\nLANGUAGE: Respond entirely in {language}."
    
    astro_context = format_astro_payload(astro_payload)
    
    user_content = f"""Here is the astrological context:

{astro_context}

User's question: {user_question}

Please provide thoughtful cosmic guidance based on the astrological data above."""
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def format_astro_payload(payload: dict) -> str:
    lines = []
    
    if "natal" in payload:
        natal = payload["natal"]
        lines.append("=== NATAL CHART ===")
        if "planets" in natal:
            lines.append("\nNatal Planets:")
            for planet, data in natal["planets"].items():
                lines.append(f"  - {planet}: {data.get('sign', 'Unknown')} ({data.get('deg_in_sign', 0):.1f})")
        if "houses" in natal:
            houses = natal["houses"]
            lines.append(f"\nAscendant: {houses.get('asc', 'Unknown')}")
            lines.append(f"Midheaven: {houses.get('mc', 'Unknown')}")
    
    if "transits" in payload:
        transits = payload["transits"]
        lines.append("\n=== CURRENT TRANSITS ===")
        if "planets" in transits:
            lines.append("\nTransit Planets:")
            for planet, data in transits["planets"].items():
                lines.append(f"  - {planet}: {data.get('sign', 'Unknown')} ({data.get('deg_in_sign', 0):.1f})")
    
    if "aspects" in payload:
        aspects = payload["aspects"]
        if aspects:
            lines.append("\n=== KEY ASPECTS (Transit to Natal) ===")
            for asp in aspects[:10]:
                lines.append(
                    f"  - Transit {asp['transit_planet']} {asp['aspect']} Natal {asp['natal_planet']} "
                    f"(orb: {asp['orb']}) - {asp['influence']}"
                )
    
    return "\n".join(lines) if lines else "No astrological data provided."
