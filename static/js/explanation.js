let explChart = null;

function renderExplanation(factors) {
    if (!factors || factors.length === 0) return;

    // Chart
    const ctx = document.getElementById('explanationChart').getContext('2d');

    if (explChart) {
        explChart.destroy();
    }

    explChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: factors.map(f => f.feature),
            datasets: [{
                data: factors.map(f => f.impact),
                backgroundColor: factors.map(f =>
                    f.direction === 'positive' ? 'rgba(0, 255, 136, 0.7)' : 'rgba(255, 71, 87, 0.7)'
                ),
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `Impact: ${context.raw > 0 ? '+' : ''}${context.raw.toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Impact on Probability', color: '#8b949e', font: { size: 11 } },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#8b949e', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#c9d1d9', font: { size: 11 } }
                }
            }
        }
    });

    // List context
    const ul = document.getElementById('explanationFactors');
    ul.innerHTML = '';
    factors.forEach(f => {
        const li = document.createElement('li');
        const icon = f.direction === 'positive' ? '<span style="color: #4ade80;">↑</span>' : '<span style="color: #f87171;">↓</span>';
        li.innerHTML = `${icon} <strong>${f.feature}</strong> changed probability by ${(Math.abs(f.impact) * 10).toFixed(1)}%`;
        li.style.background = 'rgba(255,255,255,0.03)';
        li.style.padding = '8px 12px';
        li.style.borderRadius = '6px';
        ul.appendChild(li);
    });
}
