
document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('#dataForm');
    
    const resultDiv = document.querySelector('.result');
    const errorSpan = document.querySelector('.error');

    form.addEventListener('submit', async (event) => {


        
       
        event.preventDefault();

        errorSpan.classList.add('hide');

        
        if (!form.checkValidity()) {
            form.reportValidity();
            errorSpan.classList.remove('hide');
            return;
        }

        
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                
                resultDiv.innerHTML = `<h3>Prediction Result</h3>
                                       <h3><strong>${result.message}</strong></h3>
                                        `;
                resultDiv.classList.remove('hide');
            } else {
                
                alert(`Error: ${result.error || 'An unknown error occurred.'}`);
            }
        } catch (err) {
            
            console.error('Fetch Error:', err);
            alert('Something goes wrong.');
        }
    });
});

// navigation function

const hamburger = document.querySelector('.bar');
    const navMenu = document.querySelector('nav');

    hamburger.addEventListener('click', () => {
        navMenu.classList.toggle('active');
    });