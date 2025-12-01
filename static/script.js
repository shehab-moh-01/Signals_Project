document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');
    if (form) {
        form.addEventListener('submit', (e) => {
            const fileInput = form.querySelector('input[type="file"]');
            if (!fileInput.files.length) {
                // Allow form submission without file to use sample audio
                return;
            }
            const file = fileInput.files[0];
            if (!file.name.endsWith('.wav')) {
                e.preventDefault();
                alert('Please upload a WAV file.');
            }
        });
    }

    // Add click animation to buttons
    const buttons = document.querySelectorAll('button, a');
    buttons.forEach(button => {
        button.addEventListener('click', () => {
            button.style.transform = 'scale(0.95)';
            setTimeout(() => {
                button.style.transform = 'scale(1)';
            }, 100);
        });
    });
});