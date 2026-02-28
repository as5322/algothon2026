// the following is tampermonkey code

// ==UserScript==
// @name         New Userscript
// @namespace    http://tampermonkey.net/
// @version      2024-11-16
// @description  try to take over the world!
// @author       You
// @match        https://docs.google.com/forms/d/e/*/viewform*
// @match        https://docs.google.com/forms/u/0/d/e/*/formResponse
// @match        https://docs.google.com/forms/u/0/d/e/*/formResponse?pli=1
// @icon         https://www.google.com/s2/favicons?sz=64&domain=google.com
// @grant        GM_xmlhttpRequest
// ==/UserScript==

(function () {
    'use strict';

    // Helper function to reload after 19 minutes
    function reloadAfterDelay() {
        console.log('Waiting 19 minutes before reloading...');
        setTimeout(() => {
            console.log('Reloading now...');
            window.location.href = 'https://docs.google.com/forms/d/e/1FAIpQLSeUYMkI5ce18RL2aF5C8I7mPxF7haH23VEVz7PQrvz0Do0NrQ/viewform';  // Reload original form URL
        }, 1 * 60 * 1000); // 19 minutes in milliseconds
    }

    // Check if we're on the "Thank You" page (after form submission)
    if (window.location.href.includes('/formResponse')) {
        console.log('Form submitted, now on "Thank You" page.');
        reloadAfterDelay(); // Reload after 19 minutes from "Thank You" page
        return; // Stop further execution since we're already done with submission
    }
})();

// ==UserScript==
// @name         New Userscript
// @namespace    http://tampermonkey.net/
// @version      2024-11-16
// @description  try to take over the world!
// @author       You
// @match        https://docs.google.com/forms/d/e/*/viewform*
// @match        https://docs.google.com/forms/d/e/*/formResponse*
// @icon         https://www.google.com/s2/favicons?sz=64&domain=google.com
// @grant        GM_xmlhttpRequest
// ==/UserScript==

(function () {
    'use strict';

    // Helper function to reload after 19 minutes
    function reloadAfterDelay() {
        console.log('Waiting 19 minutes before reloading...');
        setTimeout(() => {
            console.log('Reloading now...');
            window.location.href = 'https://docs.google.com/forms/d/e/1FAIpQLSeUYMkI5ce18RL2aF5C8I7mPxF7haH23VEVz7PQrvz0Do0NrQ/viewform';  // Reload original form URL
        }, 10 * 1000); // 19 minutes in milliseconds
    }

    // Check if we're on the "Thank You" page (after form submission)
    if (window.location.href.includes('/formResponse')) {
        console.log('Form submitted, now on "Thank You" page.');
        reloadAfterDelay(); // Reload after 19 minutes from "Thank You" page
        return; // Stop further execution since we're already done with submission
    }

    // Wait for the page to fully load
    window.addEventListener('load', () => {
        console.log('Page loaded, starting automation...');

        // Select and click the checkbox
        const checkboxXPath = '/html/body/div/div[2]/form/div[2]/div/div[2]/div[1]/div[1]/label/div/div[1]';
        const checkbox = document.evaluate(
            checkboxXPath,
            document,
            null,
            XPathResult.FIRST_ORDERED_NODE_TYPE,
            null
        ).singleNodeValue;

        if (checkbox) {
            checkbox.click();
            console.log('Checkbox clicked.');
        } else {
            console.error('Checkbox not found.');
        }

        // Fetch data from an API
        GM_xmlhttpRequest({
            method: 'GET',
            url: 'http://127.0.0.1:5000',  // Replace with your actual API URL
            onload: (response) => {
                const responseText = response.responseText; // Handle response text or JSON

                // Enter the response into the textarea
                const textareaXPath =
                    '/html/body/div/div[2]/form/div[2]/div/div[2]/div[2]/div/div/div[2]/div/div[1]/div[2]/textarea';
                const textarea = document.evaluate(
                    textareaXPath,
                    document,
                    null,
                    XPathResult.FIRST_ORDERED_NODE_TYPE,
                    null
                ).singleNodeValue;

                if (textarea) {
                    textarea.value = responseText;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    console.log('Response entered.');
                } else {
                    console.error('Textarea not found.');
                }

                // Wait 30 seconds before submitting the form (for testing)
                console.log('Waiting 30 seconds before submitting...');
                setTimeout(() => {
                    // Click the submit button
                    const submitXPath = '/html/body/div/div[2]/form/div[2]/div/div[3]/div[1]/div[1]/div';
                    const submitButton = document.evaluate(
                        submitXPath,
                        document,
                        null,
                        XPathResult.FIRST_ORDERED_NODE_TYPE,
                        null
                    ).singleNodeValue;

                    if (submitButton) {
                        submitButton.click();
                        console.log('Form submitted.');

                        // Wait for submission to complete and then reload after 19 minutes
                        reloadAfterDelay(); // Reload after submission

                    } else {
                        console.error('Submit button not found.');
                    }
                }, 10000); // Wait for 30 seconds before submitting the form

            },
            onerror: () => {
                console.error('Failed to fetch data from the API.');
            },
        });
    });
})();