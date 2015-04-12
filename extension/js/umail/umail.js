// Global Gmail.js object
var gmail;

// Main function for setup
var uMailMain = function() {
    gmail = Gmail();
    gmail.observe.on("open_email", function(id) { emailOpened(id); });  
};

// Callback for when an email is opened
var emailOpened = function(id) {
  var emailData = gmail.get.email_data(id);
  console.log(emailData);
  addUMailButton();
};

// Add a uMail button to the email view toolbar
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
    setTimeout(function() {
      $('.ii.gt.m14bd7f29d8697d00.adP.adO').html('<div id=":1tg" class="ii gt m14bd7f29d8697d00 adP adO"><div id=":1tf" class="a3s" style="overflow: hidden;"><div dir="ltr"><div style="font-size:12.8000001907349px">My fellow students,</div><div style="font-size:12.8000001907349px"><br></div><div style="font-size:12.8000001907349px">I am excited to announce we are approaching the election season of the&nbsp;<a href="tel:2015-2016" value="+85220152016" target="_blank">2015-2016</a>&nbsp;Student Government at NYU Shanghai.</div><div style="font-size:12.8000001907349px"><br></div><div style="font-size:12.8000001907349px">I would like to invite you to our&nbsp;<b><a href="https://orgsync.com/86544/events/1013836/occurrences/2237638" target="_blank">Elections Informational Meeting</a> on&nbsp;<span><span>March 5th</span></span>, during the&nbsp;<span><span><span class="aBn" data-term="goog_1171584949" tabindex="0"><span class="aQJ">12:30-1:45</span></span></span></span>&nbsp;lunch hour (room TBD)</b>At this meeting, the Student Government and the Elections Board will be present to provide information and answer questions about the election timeline, candidacy, and campaign rules and regulations.</div><div style="font-size:12.8000001907349px"><br></div><div style="font-size:12.8000001907349px">This year\'s elections will be notably different from previous elections as the current Executive Board has been working determinedly to revise the Student Constitution and structure of Student Government.&nbsp;</div><div style="font-size:12.8000001907349px"><br></div><div style="font-size:12.8000001907349px"><b>The Student Constitution will be open to you for comment until March. 5. After you review the Constitution, please vote to ratify the revisions&nbsp;<a href="https://orgsync.com/86544/forms/136886" target="_blank">here on OrgSync.</a>&nbsp;</b>If you have comments on the Constitution, please email&nbsp;<span style="font-size:12.8000001907349px"><b><a>shanghai.student.<wbr>government@nyu.edu</a>.</b></span></div><div style="font-size:12.8000001907349px"><br></div><div style="font-size:12.8000001907349px">Please&nbsp;<a href="https://orgsync.com/86544/events/1013836/occurrences/2237638" target="_blank">RSVP here</a>&nbsp;if you plan to attend the Elections Informational Meeting.</div><div style="font-size:12.8000001907349px"><br></div><div style="font-size:12.8000001907349px">​<span><div class="gmail_chip gmail_drive_chip" style="width:396px;min-height:18px;max-height:18px;background-color:#f5f5f5;padding:5px;color:#222;font-family:arial;font-style:normal;font-weight:bold;font-size:13px;border:1px solid #ddd;line-height:1"><a href="https://docs.google.com/a/nyu.edu/document/d/15yet-Q8oBHgZwkHucGLZzA7osHIe4F4eNl2n61WfRuk/edit?usp=drive_web" style="display:inline-block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-decoration:none;padding:1px 0px;border:none;width:100%" target="_blank"><img style="vertical-align:bottom;border:none" src="https://ci3.googleusercontent.com/proxy/Z0ycuz1AyZr8d50VlAiStPDeDqiElh0cX1NWybBlZkEsBHrBYiIl9Z8Y7ikca3A0lTDwCt7CL1B5O4Q5_cszgf-Ie1LOhRRGRg4GV17yMhzaRyyAPeh2HUAeWA=s0-d-e1-ft#https://ssl.gstatic.com/docs/doclist/images/icon_11_document_list.png" class="CToWUd">&nbsp;<span dir="ltr" style="color:#15c;text-decoration:none;vertical-align:bottom">Student Constitution 2015-2016</span></a></div>&nbsp;</span><br></div><div style="font-size:12.8000001907349px">​<span><div class="gmail_chip gmail_drive_chip" style="width:396px;min-height:18px;max-height:18px;background-color:#f5f5f5;padding:5px;color:#222;font-family:arial;font-style:normal;font-weight:bold;font-size:13px;border:1px solid #ddd;line-height:1"><a href="https://docs.google.com/a/nyu.edu/document/d/1qgR8C9zN30x_LO0CKskpG5BFBu0EaIgJ_WlpVizjNQw/edit?usp=drive_web" style="display:inline-block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-decoration:none;padding:1px 0px;border:none;width:100%" target="_blank"><img style="vertical-align:bottom;border:none" src="https://ci3.googleusercontent.com/proxy/Z0ycuz1AyZr8d50VlAiStPDeDqiElh0cX1NWybBlZkEsBHrBYiIl9Z8Y7ikca3A0lTDwCt7CL1B5O4Q5_cszgf-Ie1LOhRRGRg4GV17yMhzaRyyAPeh2HUAeWA=s0-d-e1-ft#https://ssl.gstatic.com/docs/doclist/images/icon_11_document_list.png" class="CToWUd">&nbsp;<span dir="ltr" style="color:#15c;text-decoration:none;vertical-align:bottom">NYU Shanghai Constitution.docx</span></a></div>&nbsp;</span>​</div><div style="font-size:12.8000001907349px"><br></div><div style="font-size:12.8000001907349px">Best wishes,</div><div style="font-size:12.8000001907349px">Roxanne Roman, President</div><div style="font-size:12.8000001907349px">On Behalf of Student Government</div></div><div class="yj6qo ajU"><div id=":1un" class="ajR" role="button" tabindex="0" data-tooltip="Show trimmed content" aria-label="Show trimmed content"><img class="ajT" src="//ssl.gstatic.com/ui/v1/icons/mail/images/cleardot.gif"></div></div><span class="HOEnZb adL"><font color="#888888"><p></p>-- <br>Responses to this email should be directed to the sender. Please do not write emails to <a href="mailto:shanghai-co17-group@nyu.edu" target="_blank">shanghai-co17-group@nyu.edu</a>, as they may not reach their intended destination.<br></font></span></div></div>');
    }, 800);
  });

  var content = $(document.createElement('div'));
  content.attr('class','asa');

  buttonContainer.html(button);
  toolbar.append(buttonContainer);
}

// Initial check that jQuery and Gmail.js is loaded
var checkLoaded = function() {
  if(window.jQuery && window.Gmail) { uMailMain(); } 
  else { setTimeout(checkLoaded, 100); }
};

checkLoaded();