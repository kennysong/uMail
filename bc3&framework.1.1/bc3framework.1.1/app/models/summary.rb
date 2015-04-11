class Summary < ActiveRecord::Base
  belongs_to :email_thread
  belongs_to :experiment
  
  #sum
  def sum_sentences
    if self.Sum.blank?
      []
    else
      #Correct multiple boxes
      self.Sum.sub!(/\]\s*\[/,',')
      a = self.Sum.split(/\s*\[(?:\d|\.|,|\s)+\](?:\s|,|\.)*/) 
      b = self.Sum.scan(/\s*\[(?:\d|\.|,|\s)+\](?:\s|,|\.)*/).map do |element|
        res = element.match(/(\d+\.\d+\s*,)*\s*(\d+\.\d+)/)
        if res.nil?
          element = ""
        else
          element = res[0]
        end
      end
      a.zip(b)
    end
  end
  #sent
  def sentences
    if self.Sent.blank?
      []
    else
      self.Sent.split(',').map do |entry|
        entry = entry[1..-1]
      end
    end
  end
  #label
  def labels
    res = {"meet" => [], "meta" => [], "req" => [], "prop" => [], "cmt" => [], "subj" => []}
    unless self.Label.blank?
      self.Label.split(',').map do |entry|
        if entry =~ /(\D+)(\d+\.\d+)/
          res[$1] << $2
        end
      end
    end
    res
  end
end
