function createArc(cx, cy, r, startAngle, endAngle, color) {
    const start = polarToCartesian(cx, cy, r, endAngle);
    const end = polarToCartesian(cx, cy, r, startAngle);
    const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
    
    const d = [
        "M", start.x, start.y, 
        "A", r, r, 0, largeArcFlag, 0, end.x, end.y
    ].join(" ");

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", color);
    path.setAttribute("stroke-width", "16");
    path.setAttribute("stroke-linecap", "round");
    return path;
}

function polarToCartesian(centerX, centerY, radius, angleInDegrees) {
    const angleInRadians = (angleInDegrees - 180) * Math.PI / 180.0;
    return {
        x: centerX + (radius * Math.cos(angleInRadians)),
        y: centerY + (radius * Math.sin(angleInRadians))
    };
}

function getColor(probability) {
    if (probability < 0.6) return '#facc15'; // yellow (toss-up)
    if (probability < 0.7) return '#86efac'; // light green
    if (probability < 0.8) return '#4ade80'; // green
    if (probability < 0.9) return '#16a34a'; // dark green
    return '#22c55e'; // bright green
}

function createGauge(containerId, probability, color1, color2) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 200 120");
    svg.setAttribute("class", "gauge-svg");
    svg.style.width = "100%";
    svg.style.maxWidth = "250px";
    
    // Background arc
    const bgArc = createArc(100, 100, 80, 0, 180, '#2a2a3e');
    svg.appendChild(bgArc);
    
    // Determine the angle sweep (180 to 0 because SVG angles are inverted here)
    // probability goes from 0.5 (center) to 1.0 (right side) for winner
    const targetAngle = 180 * probability;
    
    const probArc = createArc(100, 100, 80, 180 - targetAngle, 180, getColor(probability));
    
    // Animation via stroke-dasharray
    const circumference = Math.PI * 80;
    probArc.style.strokeDasharray = circumference;
    probArc.style.strokeDashoffset = circumference;
    probArc.style.transition = 'stroke-dashoffset 1.5s ease-out';
    svg.appendChild(probArc);
    
    // Center text
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", "100");
    text.setAttribute("y", "90");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("fill", "#fff");
    text.setAttribute("font-size", "24");
    text.setAttribute("font-weight", "bold");
    text.textContent = `${(probability * 100).toFixed(1)}%`;
    svg.appendChild(text);
    
    document.getElementById(containerId).appendChild(svg);
    
    // Trigger animation
    setTimeout(() => {
        probArc.style.strokeDashoffset = circumference - (circumference * (targetAngle / 180));
    }, 50);
}
