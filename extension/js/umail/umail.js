// Global Gmail.js object
var gmail;

// Global variables for current email
var length_ratio = 0.5;
var email_id;
var sentences_sorted;
var sentence_index;
var processed_sentence_to_original;
var email_body;
var ss;

// Initial check that jQuery and Gmail.js is loaded
var checkLoaded = function() {
  if(window.jQuery && window.Gmail) { uMailMain(); } 
  else { setTimeout(checkLoaded, 100); }
};

// Main function for setup
var uMailMain = function() {
  gmail = Gmail();
  gmail.observe.on("open_email", function(id) { emailOpened(id); });  
};

// Callback for when an email is opened
var emailOpened = function(id) {
  // Get email data of first email in thread
  var emailData = gmail.get.email_data(id);
  var firstEmail = emailData.threads[id]; // ID of thread == ID of first email

  // Remove all HTML tags, but keep newlines 
  var emailHTML = firstEmail.content_html;
  emailHTML = emailHTML.replace(/<br ?\\?>|<\/div>|<\/p>/g, "\n")
  emailHTML = emailHTML.replace(/<(?:"[^"]*"['"]*|'[^']*'['"]*|[^'">])+>/g, ""); // Wow!

  // Convert into a query string to POST to server
  var dataString = "email=" + encodeURIComponent(emailHTML) +
                   "&subject=" + encodeURIComponent(firstEmail.subject) +
                   "&to=" + encodeURIComponent(firstEmail.to) +
                   "&cc=" + encodeURIComponent(firstEmail.cc);

  // Send data to server on email load (rather than on button click) to reduce latency
  window.postMessage({type: 'new_email_request', data: dataString}, '*');

  // Record email ID as a global variable
  email_id = id;

  // Add Summarize button to email view toolbar
  addUMailButton();
};

// Add a Summarize button to the email view toolbar
// Adapted from api.tools.add_toolbar_button() in Gmail.js
var addUMailButton = function() {
  var toolbar = $("[gh='mtb']");
  while(!toolbar.hasClass('iH')) {
    // Hack to get the email view toolbar, not the main toolbar
    setTimeout(addUMailButton, 500);
    return;
  }
  while(toolbar.children().length == 1) {
    toolbar = $(toolbar).children().first();
  }

  var buttonContainer = $(document.createElement('div'));
  buttonContainer.attr('class','G-Ni J-J5-Ji');

  var button = $(document.createElement('div'));
  button.attr('class', 'T-I J-J5-Ji lS T-I-ax7 ar7');

  button.html('<div><img src="' + uMailExtensionURL + 
    'img/iconToolbar.png" style="position: relative;top: 4px;"> Summarize</div>');

  button.click(function() {
    displaySummary();
  });

  var content = $(document.createElement('div'));
  content.attr('class','asa');

  buttonContainer.html(button);
  toolbar.append(buttonContainer);
}

// Display summary for current email
var displaySummary = function() {
  // Take # num_sentences of the most important sentences, 
  // keeping track of index in original email
  var num_sentences = sentences_sorted.length * length_ratio;
  var summary = [];
  for (var i = 0; i < num_sentences; i++)  {
    var current_sentence = sentences_sorted[i];
    summary.push({
                  'sent': current_sentence,
                  'index': sentence_index[processed_sentence_to_original[current_sentence]]
                });
  }
  var summary = sortByKey(summary, 'index');

  // Turn this summary array into a string, replacing \n with <br>
  var summaryStr = '';
  for (var i = 0; i < summary.length; i++) {
    var current_sentence = summary[i]['sent'];
    current_sentence = current_sentence.replace(/\n/g, "<br>");
    summaryStr += current_sentence + ' ';
  }

  // Display summary on page
  var emailElement = new gmail.dom.email(email_id);
  emailElement.body(summaryStr);
}

// Utility function to sort a array of objects by a key
function sortByKey(array, key) {
    return array.sort(function(a, b) {
        var x = a[key]; var y = b[key];
        return ((x < y) ? -1 : ((x > y) ? 1 : 0));
    });
}

// Add a message listener for communication with content.js
window.addEventListener('message', function(event) {
    // Listener for response from /new_email
    if (event.data.type == 'new_email_response') {
        console.log(event.data.data)
        var summaryJSON = JSON.parse(event.data.data);
        sentences_sorted = summaryJSON['sent_sorted'];
        sentence_index = summaryJSON['sent_index'];
        processed_sentence_to_original = summaryJSON['processed_sent_to_original'];
    }
});

// Get ready to call uMailMain()
checkLoaded();