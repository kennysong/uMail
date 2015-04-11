class Email < ActiveRecord::Base
  belongs_to :email_thread
  def email_file=(file_data)
      @file_data = file_data
  end
  
  def sentences
    #Remove leading and trailing ^
    text = self.Body.reverse.chop.reverse.chop
    #Split along ^\n^
    text.split(/\^\r\n\^|\^\^/)
  end
end
